from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

# Tải mô hình đã lưu
model = joblib.load("svm_model.pkl")

# Khởi tạo ứng dụng với thông tin 100% tiếng Việt
app = FastAPI(
    title="Hệ Thống Phân Loại Hoa Iris 🌺",
    description="Ứng dụng máy học SVM dự đoán giống hoa Iris.",
    version="1.0.0",
    docs_url="/tai-lieu", # Đổi link tài liệu từ /docs sang /tai-lieu
)

# Cấu trúc dữ liệu nhập vào (Việt hóa)
class IrisInput(BaseModel):
    sepal_length: float = Field(..., title="Chiều dài đài hoa (cm)", example=5.1)
    sepal_width: float = Field(..., title="Chiều rộng đài hoa (cm)", example=3.5)
    petal_length: float = Field(..., title="Chiều dài cánh hoa (cm)", example=1.4)
    petal_width: float = Field(..., title="Chiều rộng cánh hoa (cm)", example=0.2)

species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Endpoint xử lý dự đoán
@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    return {
        "class_id": prediction,
        "prediction": species[prediction]
    }

# Endpoint giao diện chính (Trang Web đẹp)
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Phân Loại Hoa Iris 🌺</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .bg-custom { background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); }
        </style>
    </head>
    <body class="bg-custom flex items-center justify-center min-h-screen">
        <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md">
            <div class="text-center mb-8">
                <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-500 to-emerald-700">
                    🌺 Trợ Lý Phân Loại Iris
                </h1>
                <p class="text-gray-500 mt-2 text-sm">Nhập thông số (cm) để máy học dự đoán</p>
            </div>
            
            <div class="space-y-5">
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Chiều dài đài hoa (Sepal Length)</label>
                    <input type="number" id="sl" value="5.1" step="0.1" class="w-full border-2 border-gray-200 p-3 rounded-xl focus:outline-none focus:border-green-500 transition">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Chiều rộng đài hoa (Sepal Width)</label>
                    <input type="number" id="sw" value="3.5" step="0.1" class="w-full border-2 border-gray-200 p-3 rounded-xl focus:outline-none focus:border-green-500 transition">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Chiều dài cánh hoa (Petal Length)</label>
                    <input type="number" id="pl" value="1.4" step="0.1" class="w-full border-2 border-gray-200 p-3 rounded-xl focus:outline-none focus:border-green-500 transition">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Chiều rộng cánh hoa (Petal Width)</label>
                    <input type="number" id="pw" value="0.2" step="0.1" class="w-full border-2 border-gray-200 p-3 rounded-xl focus:outline-none focus:border-green-500 transition">
                </div>
                
                <button onclick="duDoanHoa()" class="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold py-3 px-4 rounded-xl hover:shadow-lg transform hover:-translate-y-0.5 transition duration-200 text-lg mt-4">
                    🔍 Dự Đoán Kết Quả
                </button>
            </div>

            <div id="ketqua-box" class="mt-6 p-4 rounded-xl bg-green-50 border border-green-200 hidden text-center">
                <p class="text-sm text-green-800 font-medium">Giống hoa được dự đoán là:</p>
                <p id="ketqua-text" class="text-2xl font-extrabold text-green-600 mt-1"></p>
            </div>
        </div>

        <script>
            async function duDoanHoa() {
                const data = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                };
                
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await res.json();
                    
                    document.getElementById('ketqua-box').classList.remove('hidden');
                    document.getElementById('ketqua-text').innerText = "🌸 " + result.prediction;
                } catch (error) {
                    alert("Lỗi kết nối đến máy chủ API!");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)