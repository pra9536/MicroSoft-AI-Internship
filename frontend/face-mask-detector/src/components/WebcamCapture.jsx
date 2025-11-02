import axios from "axios";
import { useRef, useState } from "react";
import Webcam from "react-webcam";
import "./WebcamCapture.css";

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [label, setLabel] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const captureAndPredict = async () => {
    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) {
      setError("❗ Camera not ready. Please try again.");
      return;
    }

    setLoading(true);
    setError("");
    setLabel("");

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        { image: imageSrc },
        { timeout: 5000 } // 5 seconds timeout
      );

      setLabel(response.data.label);
    } catch (err) {
      console.error("Prediction error:", err);

      if (err.code === "ECONNABORTED") {
        setError("⏳ Server took too long to respond. Please try again later.");
      } else if (err.response) {
        setError("⚠️ Server error occurred. Please try again later.");
      } else {
        setError("🚫 Unable to connect to server. Please ensure backend is running.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">😷 Face Mask Detection</h1>

      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
      />

      <div className="button-group">
        <button className="btn" onClick={captureAndPredict} disabled={loading}>
          {loading ? "Processing..." : "Predict Mask"}
        </button>

        {error && (
          <button className="btn retry" onClick={captureAndPredict}>
            🔁 Retry
          </button>
        )}
      </div>

      {label && (
        <h2 className={`result ${label === "Mask" ? "mask" : "no-mask"}`}>
          {label === "Mask" ? "✅ Mask Detected" : "❌ No Mask Detected"}
        </h2>
      )}

      {error && <h3 className="error">{error}</h3>}
    </div>
  );
};

export default WebcamCapture;
