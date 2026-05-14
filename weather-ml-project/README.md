cd C:\Users\tzoan\Desktop\Weather_Forecasting\weather-ml-project
# create + activate venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# run full pipeline
python3 main.py


Run the frontend (web app)
cd C:\Users\tzoan\Desktop\Weather_Forecasting\frontend
npm install
npm run dev
Then open the local URL Vite prints (usually http://localhost:5173).
does github work ?