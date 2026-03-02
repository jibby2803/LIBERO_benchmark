import os 
import sys 
import math
import numpy as np 
import torch

from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors


class Pi0_inference():
    def __init__(self, model_dir="", infer_chunk=10):
        self.model_dir = model_dir 
        self.device = 'cuda'
        self.infer_chunk=infer_chunk
        self.policy = self.create_policy()
        
        self.preprocessor, self.postprocessor = make_pre_post_processors(  
            policy_cfg=self.policy.config,  
            pretrained_path=self.model_dir,  
            dataset_stats=None  # Load from model if needed  
        )  
        

    def create_policy(self):
        try: 
            policy = PI0Policy.from_pretrained(self.model_dir)
            policy = policy.to(self.device)
            return policy 
        except Exception as e:
            print(f"❌ Failed to load policy: {e}")
            sys.exit(1)
            
    
    def get_libero_action(self, obs, task_description):
        data = self._process_observation(obs, task_description)
        batch: dict[str, torch.Tensor] = {}
        state = torch.as_tensor(data["observation.state"], dtype=torch.float32, device=self.device)
        batch["observation.state"] = state.unsqueeze(0)
        
        for key in ("observation.images.image", "observation.images.image2"):
            img = torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):  # HWC → CHW
                img = img.permute(2, 0, 1)
            img = img[:3, :, :]  # keep only RGB if 4-channel (e.g. RGBA)
            img = img / 255.0    # scale to [0,1]
            batch[key] = img.unsqueeze(0)
        batch["task"] = [data["task"]]
        
        batch = self.preprocessor(batch)
        with torch.no_grad():
            # action = self.client.select_action_chunk(batch)
            # print(batch)
            action = self.policy.predict_action_chunk(batch)
            action = self.postprocessor(action)
            action = action.cpu().numpy()
        action = action[0]
        action = self.normalize_gripper_action(action)
        # print(action)
        # print(action.shape)
        return action[:min(len(action), self.infer_chunk)] 
    
    def _process_observation(self, obs, task_description):

        element = {
            "observation.images.image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "observation.images.image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            "observation.state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "task": str(task_description),
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