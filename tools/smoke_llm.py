import traceback
from pathlib import Path

def run():
    try:
        from nanovllm.engine.llm_engine import LLMEngine
        from nanovllm.sampling_params import SamplingParams
    except Exception as e:
        print('import error:', type(e).__name__, e)
        traceback.print_exc()
        return
    model_dir = Path(__file__).resolve().parent.parent / 'test_model'
    print('Model dir:', model_dir)
    try:
        engine = LLMEngine(str(model_dir), tensor_parallel_size=1, enforce_eager=True)
        sp = SamplingParams()
        gen = engine.stream_generate('Hello', sp)
        print('engine instantiated; streaming generator created')
        for i, chunk in enumerate(gen):
            print('chunk', i, repr(chunk))
            if i > 5:
                break
    except Exception as e:
        print('instantiation/generation error:', type(e).__name__, e)
        traceback.print_exc()

if __name__ == '__main__':
    run()
