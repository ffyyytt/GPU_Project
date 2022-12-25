# GPU_Project
[Google Colab](https://colab.research.google.com/drive/1KvNqcUhIPcUWGdpuyPGIak-UFbXNB3QI?usp=sharing)
## Chateau
![Chateau](https://github.com/ffyyytt/GPU_Project/raw/main/report/images/report_chateau.png)
## Palais
![Palais](https://github.com/ffyyytt/GPU_Project/raw/main/report/images/report_palais.png)
## Lenna
```python
image = cv2.cvtColor(cv2.imread("images/Lenna.png"), cv2.COLOR_BGR2RGB)
image_brightness = change_brightness(image, -120)
cv2.imwrite("images/Lenna_brightness.png", cv2.cvtColor(image_brightness, cv2.COLOR_RGB2BGR))
```
![Lenna changed](https://github.com/ffyyytt/GPU_Project/raw/main/report/images/lenna_changed.png)
![Lenna](https://github.com/ffyyytt/GPU_Project/raw/main/report/images/report_lenna.png)
