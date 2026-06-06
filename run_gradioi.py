import os
from io import BytesIO
from PIL import Image, ImageDraw

try:
    import gradio as gr
except Exception:
    gr = None

try:
    from roboflow import Roboflow
except Exception:
    Roboflow = None

API_KEY = os.getenv("ROBFLOW_API_KEY", "YOUR_API_KEY")

class MockModel:
    def predict(self, image_input, confidence=40):
        class R:
            def json(self):
                return {"predictions":[{"class":"drill","width":120.5,"height":45.2,"confidence":0.93}]}
        return R()

model = None
if API_KEY and API_KEY != "YOUR_API_KEY" and Roboflow is not None:
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace("witeks-workspace").project("dekracoating")
        model = project.version(4).model
    except Exception as e:
        print("Roboflow init failed:", e)
        model = MockModel()
else:
    model = MockModel()


def scan_drill(image):
    if image is None:
        return {"error": "No image provided"}
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    try:
        result = model.predict(buf, confidence=40).json()
    except Exception:
        image.save("temp.jpg")
        result = model.predict("temp.jpg", confidence=40).json()
    predictions = []
    for pred in result.get("predictions", []):
        predictions.append({
            "type": pred.get("class"),
            "width_px": round(pred.get("width", 0), 1),
            "height_px": round(pred.get("height", 0), 1),
            "confidence": round(pred.get("confidence", 0), 2)
        })
    return predictions


demo = None
if gr is not None:
    demo = gr.Interface(
        fn=scan_drill,
        inputs=gr.Image(type="pil", label="Upload a drill photo"),
        outputs=gr.JSON(label="Drill info"),
        title="Drill Scanner – Dekracoating"
    )


if __name__ == "__main__":
    # quick self-test without launching Gradio
    test_img = Image.new("RGB", (200, 200), color=(255,255,255))
    d = ImageDraw.Draw(test_img)
    d.rectangle([50,50,150,100], outline="black")
    print("Self-test output:", scan_drill(test_img))
    if os.getenv("RUN_GRADIO") == "1":
        if demo is None:
            raise RuntimeError("Gradio is not installed; cannot launch UI.")
        demo.launch()
