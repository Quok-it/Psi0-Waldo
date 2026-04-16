import os
import sys
import threading
import time
from enum import IntEnum
from multiprocessing import Array, Event, Lock, Process, shared_memory

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

brainco_Num_Motors = 6
kTopicBraincoLeftCommand = "rt/brainco/left/cmd"
kTopicBraincoRightCommand = "rt/brainco/right/cmd"
kTopicBraincoLeftState = "rt/brainco/left/state"
kTopicBraincoRightState = "rt/brainco/right/state"


class Brainco_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5


class Brainco_Right_Hand_JointIndex(IntEnum):
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5


class Brainco_Inference_Controller:
    """BrainCo Revo2 hand controller for inference.

    Drop-in replacement for Dex3_1_Controller. Reads 12 floats from
    hand_shm_array (6 left + 6 right), publishes to BrainCo DDS topics,
    and exposes the same interface that master_whole_body.py expects.
    """

    def __init__(
        self,
        hand_shm_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
    ):
        print("Initialize Brainco_Inference_Controller...")
        self.fps = fps

        # DDS publishers
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicBraincoLeftCommand, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicBraincoRightCommand, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        # DDS subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicBraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicBraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        # Initialize command messages
        self.left_msg = MotorCmds_()
        self.left_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]
        for idx in Brainco_Left_Hand_JointIndex:
            self.left_msg.cmds[idx].q = 0.0
            self.left_msg.cmds[idx].dq = 1.0

        self.right_msg = MotorCmds_()
        self.right_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]
        for idx in Brainco_Right_Hand_JointIndex:
            self.right_msg.cmds[idx].q = 0.0
            self.right_msg.cmds[idx].dq = 1.0

        # Shared state arrays (between DDS subscriber thread and control process)
        self.left_hand_state_array = Array("d", brainco_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", brainco_Num_Motors, lock=True)

        # References for external access
        self.hand_shm_array = hand_shm_array
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array

        # Start DDS subscriber thread
        self.stop_event = Event()
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for hand state
        while not (any(self.left_hand_state_array) and any(self.right_hand_state_array)):
            time.sleep(0.01)
            print("[Brainco_Inference_Controller] Waiting to subscribe dds...")

        # Start control process
        self.hand_control_process = Process(
            target=self.control_process,
            args=(
                hand_shm_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                self.dual_hand_data_lock,
                self.dual_hand_state_array,
                self.dual_hand_action_array,
            ),
        )
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        print("Initialize Brainco_Inference_Controller OK!\n")

    def _subscribe_hand_state(self):
        while not self.stop_event.is_set():
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None:
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
            if right_hand_msg is not None:
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """Set current left, right hand motor state target q."""
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_msg.cmds[id].q = left_q_target[idx]
        self.LeftHandCmb_publisher.Write(self.left_msg)
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_msg.cmds[id].q = right_q_target[idx]
        self.RightHandCmb_publisher.Write(self.right_msg)

    def get_current_dual_hand_q(self):
        q = np.array(
            [self.left_hand_state_array[i] for i in range(brainco_Num_Motors)]
            + [self.right_hand_state_array[i] for i in range(brainco_Num_Motors)]
        )
        return q

    def get_current_dual_hand_pressure(self):
        # BrainCo Revo2 has no pressure sensors in this path
        return np.zeros(18)

    def control_process(
        self,
        hand_shm_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        while not self.stop_event.is_set():
            start_time = time.time()

            left_q_target = hand_shm_array[0:6]
            right_q_target = hand_shm_array[6:12]

            if left_q_target is not None and right_q_target is not None:
                state_data = np.concatenate(
                    (
                        np.array(left_hand_state_array[:]),
                        np.array(right_hand_state_array[:]),
                    )
                )
                action_data = np.concatenate((left_q_target, right_q_target))

                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)

            time_elapsed = time.time() - start_time
            sleep_time = max(0, (1 / self.fps) - time_elapsed)
            time.sleep(sleep_time)

        print("Brainco_Inference_Controller has been closed.")

    def shutdown(self):
        print("Shutting down Brainco_Inference_Controller...")
        self.stop_event.set()

    def reset(self, max_wait_sec=5.0):
        pass
