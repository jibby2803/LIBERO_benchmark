import os 
import sys 
import math
import numpy as np 
import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor
from eo.model.processing_eo1 import EO1VisionProcessor

class EO1_inference():
    def __init__(self, model_dir="", infer_chunk=8):
        self.model_dir = model_dir 
        self.device = 'cuda'
        self.infer_chunk=infer_chunk
        self.policy, self.processor = self.create_policy()


    def create_policy(self):
        try: 
            model = (
                AutoModel.from_pretrained(
                    self.model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    # attn_implementation="flash_attention_2",
                )
                .eval()
                .cuda()
            )

            # processor = AutoProcessor.from_pretrained(
            #     self.model_dir, trust_remote_code=True, local_files_only=True
            # )         
            processor = EO1VisionProcessor.from_pretrained(self.model_dir, trust_remote_code=True, local_files_only=True)
            
            return model, processor
        except Exception as e:
            print(f"❌ Failed to load policy: {e}")
            sys.exit(1)
            
    
    def get_libero_action(self, obs, task_description):
        data = self._process_observation(obs, task_description)
        ov_out = self.processor.select_action(
            self.policy,
            data,
        )
        action_chunk = ov_out.action[0].numpy()

        action = self.normalize_gripper_action(action_chunk)
        action = invert_gripper_action(action)
        
        action = action[: min(self.infer_chunk, len(action))]


        return action 
    
    def _process_observation(self, obs, task_description):
        
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = Image.fromarray(img)
        wrist_img = Image.fromarray(wrist_img)
        
        state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
        state = torch.from_numpy(state).float()
        
        task = str(task_description)
        
        element = {
            "observation.images.image": [img],
            "observation.images.wrist_image": [wrist_img],
            "observation.state": [state],
            "task": [task],
        }
        
        return element
    
    def normalize_gripper_action(self, action, binarize=True):
        orig_low, orig_high = 0.0, 1.0
        action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

        if binarize:
            # Binarize to -1 or +1.
            action[..., -1] = np.sign(action[..., -1])

        return action
    
    
def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action