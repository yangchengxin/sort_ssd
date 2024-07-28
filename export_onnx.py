import onnx
from nets.ssd import SSD300
import torch

def export_onnx():
    onnx_path = "ssd.onnx"
    model = SSD300(21,'vgg', False)
    model.load_state_dict(torch.load('ssd_weights.pth'))

    dummy_inputs = {
        "input" : torch.randn(1,3,300,300, dtype=torch.float),
    }

    output_names = {
        "classes", "detections"
    }

    with open(onnx_path, "wb") as f:
        print(f"exporting onnx model to {onnx_path}...")
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=11,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
        )

    

if __name__ == "__main__":
    export_onnx()