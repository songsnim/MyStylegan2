import numpy as np
import torch
import wandb
import datetime
import logging

# logger = logging.getLogger()

# logger.setLevel(logging.INFO)

# stream_handler = logging.StreamHandler()
# logger.addHandler(stream_handler)

# file_handler = logging.FileHandler("wandb_test.log")
# logger.addHandler(file_handler)

# for i in range(10):
#     logger.info(f"{i}")

today = datetime.date.today()

a = 1
b = 2

a = np.array([1, 2, 3])

c = a + 1

wandb.login

wandb.init(project="StyleGAN2", entity='songyj', name="wandb_test")

wandb.log({"a": "Importing models..."})
# wandb.log("Importing models done..!")

# wandb.log(f"오늘날짜 : {today}")

wandb.finish()
