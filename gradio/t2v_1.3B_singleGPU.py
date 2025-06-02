# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import warnings

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(
    0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

# Global Var
prompt_expander = None
wan_t2v = None


# Button Func
def prompt_enc(prompt, tar_lang):
    global prompt_expander
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt


def t2v_generation(txt2vid_prompt, resolution, sd_steps, guide_scale,
                   shift_scale, seed, n_prompt):
    global wan_t2v
    # print(f"{txt2vid_prompt},{resolution},{sd_steps},{guide_scale},{shift_scale},{seed},{n_prompt}")

    W = int(resolution.split("*")[0])
    H = int(resolution.split("*")[1])
    video = wan_t2v.generate(
        txt2vid_prompt,
        size=(W, H),
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True)

    cache_video(
        tensor=video[None],
        save_file="example.mp4",
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

    # Modified save path
    output_path = "/tmp/example.mp4"
    cache_video(
        tensor=video[None],
        save_file=output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

    return output_path


# Model Initialization Function
def initialize_models(ckpt_base_dir="/tmp/wan_cache"): # Changed default
    global prompt_expander, wan_t2v

    # Ensure the cache directory exists
    if not os.path.exists(ckpt_base_dir):
        os.makedirs(ckpt_base_dir, exist_ok=True)

    # Configuration for models (simplified, assuming 'local_qwen' and default model)
    prompt_extend_method = "local_qwen"
    prompt_extend_model = None

    device_to_use = "cpu" # Changed to CPU

    print("Step1: Init prompt_expander...", end='', flush=True)
    if prompt_extend_method == "dashscope":
        try:
            # DashScope might require API keys via environment variables
            prompt_expander = DashScopePromptExpander(
                model_name=prompt_extend_model, is_vl=False)
        except Exception as e:
            print(f"Error initializing DashScopePromptExpander: {e}")
            prompt_expander = None # Ensure it's None if failed
    elif prompt_extend_method == "local_qwen":
        try:
            prompt_expander = QwenPromptExpander(
                model_name=prompt_extend_model, is_vl=False, device=device_to_use)
        except Exception as e:
            print(f"Error initializing QwenPromptExpander on {device_to_use}: {e}")
            print("Attempting with default device (if any specified by library)")
            try:
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model, is_vl=False)
            except Exception as e_default:
                print(f"Error initializing QwenPromptExpander with default device: {e_default}")
                prompt_expander = None # Ensure it's None if failed
    else:
        print(f"Unsupport prompt_extend_method: {prompt_extend_method}")
        prompt_expander = None
    print("done" if prompt_expander else "failed", flush=True)


    print("Step2: Init 1.3B t2v model...", end='', flush=True)
    if prompt_expander is None and prompt_extend_method != "dashscope": # Qwen might be essential for some model setups
        print("Skipping WanT2V initialization due to prompt_expander failure.")
        wan_t2v = None
    else:
        cfg = WAN_CONFIGS['t2v-1.3B']
        try:
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_base_dir,
                device_id=device_to_use,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
            )
        except Exception as e:
            print(f"Error initializing WanT2V on {device_to_use}: {e}")
            print("Attempting with default device_id (if any specified by library or if it falls back)")
            try:
                wan_t2v = wan.WanT2V(
                    config=cfg,
                    checkpoint_dir=ckpt_base_dir,
                    rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False,
                )
            except Exception as e_default:
                print(f"Error initializing WanT2V with default device_id: {e_default}")
                wan_t2v = None # Ensure it's None if failed
    print("done" if wan_t2v else "failed", flush=True)


# Interface
def gradio_interface():
    # Initialize models when the interface is created
    initialize_models() # Call the new init function

    with gr.Blocks() as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (T2V-1.3B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Wan: Open and Advanced Large-Scale Video Generative Models.
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                txt2vid_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate",
                )
                tar_lang = gr.Radio(
                    choices=["ZH", "EN"],
                    label="Target language of prompt enhance",
                    value="ZH")
                run_p_button = gr.Button(value="Prompt Enhance")

                with gr.Accordion("Advanced Options", open=True):
                    resolution = gr.Dropdown(
                        label='Resolution(Width*Height)',
                        choices=[
                            '480*832',
                            '832*480',
                            '624*624',
                            '704*544',
                            '544*704',
                        ],
                        value='480*832')

                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1)
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=6.0,
                            step=1)
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift scale",
                            minimum=0,
                            maximum=20,
                            value=8.0,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add"
                    )

                run_t2v_button = gr.Button("Generate Video")

            with gr.Column():
                result_gallery = gr.Video(
                    label='Generated Video', interactive=False, height=600)

        run_p_button.click(
            fn=prompt_enc,
            inputs=[txt2vid_prompt, tar_lang],
            outputs=[txt2vid_prompt])

        run_t2v_button.click(
            fn=t2v_generation,
            inputs=[
                txt2vid_prompt, resolution, sd_steps, guide_scale, shift_scale,
                seed, n_prompt
            ],
            outputs=[result_gallery],
        )

    return demo


# Main
# The _parse_args function is removed.
# The if __name__ == '__main__': block is removed to prevent auto-launching.
# app.py will now be responsible for importing and running the Gradio app.

# if __name__ == '__main__':
#     # args = _parse_args() # Removed
#     # initialize_models(ckpt_dir=args.ckpt_dir) # Call the init function with args - logic moved
#     demo = gradio_interface()
#     # demo.launch(server_name="0.0.0.0", share=False, server_port=7860) # Removed
