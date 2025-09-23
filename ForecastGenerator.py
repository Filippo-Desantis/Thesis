## In this code we exploit Earth2Studio APIs to produce our MLWPs, we cited Earth2Studio properly in the thesis ##

import os
import logging
from datetime import datetime, timedelta
import torch
from dotenv import load_dotenv
import onnxruntime as ort

output_dir = "FCN/24hFCN"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(output_dir, 'forecast.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# model parameters
nsteps = 4  # lead time index (nsteps * 6hr)
batch_size = 1
model_name = "FCN" ## Here we can choose the MLWP model to use

# dates interval
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

# GPU setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# data management
from earth2studio.models.px.aurora import Aurora
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend


def generate_dates(start, end):
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days + 1)]

def process_dates(dates):
    for i, date in enumerate(dates):

        try:
            package = FCN.load_default_package()
            model = FCN.load_model(package).to(device)
            #model = Pangu6("/scratch3/poliMI/Filippo/Pangu-Weather/pangu_weather_24.onnx","/scratch3/poliMI/Filippo/Pangu-Weather/pangu_weather_6.onnx").to(device)
            logging.info(f"Loaded {model_name} model on {device}")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise        
        date_str = date.strftime("%Y-%m-%d")

        logging.info(f"Processing {date_str} ({i+1}/{len(dates)})")

        try:
            # memory cleaning before inference
            torch.cuda.empty_cache()
            data = GFS()            
            # output path
            output_path = os.path.join(output_dir, f"forecast_{date_str}.zarr")

            # backend
            io = ZarrBackend(output_path)

            # forecast
            from earth2studio.run import deterministic
            deterministic([date_str], nsteps, model, data, io, device=device)

            logging.info(f"Successfully saved {output_path}")

            # explicit cleaning
            del io
            del model
            del data
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed {date_str}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        all_dates = generate_dates(start_date, end_date)
        logging.info(f"Starting processing for {len(all_dates)} days")
        process_dates(all_dates)
        logging.info("Processing completed successfully")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        torch.cuda.empty_cache()

