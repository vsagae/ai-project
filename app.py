# Libraries
import tkinter as tk
from tkinter import filedialog, messagebox
from pdf2image import convert_from_path
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import pandas as pd
import re
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Função para selecionar arquivo
def selecionar_arquivo():
    global arquivo_selecionado,dir
    arquivo_selecionado = filedialog.askopenfilename()
    if arquivo_selecionado:
        label_arquivo.config(text=f"Selected: {arquivo_selecionado}")
        dir = os.path.dirname(arquivo_selecionado) 

# Função para rodar a análise
def rodar_analise():
    if not arquivo_selecionado:
        messagebox.showerror("Error", "Please, first select a file!")
        return

    messagebox.showinfo("Processing", f"Processing file:\n{arquivo_selecionado}")
    realizar_analise(arquivo_selecionado)
    messagebox.showinfo("Done", "The file was processed!")

def realizar_analise(arquivo):
    print(f"Processing {arquivo}...")
    pages = convert_from_path(arquivo, dpi=300)

    if not os.path.exists(f'{dir}/processing'):
        os.makedirs(f'{dir}/processing')
        
    for i, page in enumerate(pages):
        img = np.array(page)
        img=img
        cv2.imwrite(f"{dir}/processing/image_{i}.jpg",img)
        
    images_list=glob.glob(f"{dir}/processing/image_*.jpg")

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def noise_removal(image):
        kernel = np.ones((1,1),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1,1), np.uint8)
        image = cv2.erode(image, kernel,iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def getSkewAngle(cvImage):
        newImage=cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 0 , 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilate = cv2.dilate(thresh, kernel , iterations= 2)
        
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        for c in contours:
            rect= cv2.boundingRect(c)
            x, y, w, h=rect
            cv2.rectangle(newImage,(x,y),(x+w,y+h),(36,255,12),2)
            
        largestContour= contours[0]
        print(len(contours))
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        
        if angle < -45:
            angle = 90 + angle 
        return -1.0 * angle

    def rotateImage(cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w)= newImage.shape[:2]
        center = (w//2, h//2)
        M= cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    def deskew(cvImage):
        angle = getSkewAngle(cvImage)
        return rotateImage(cvImage, -1.0*angle)

    for i, image in enumerate(images_list):
        img = cv2.imread(image)
        cv2.imwrite(f"{dir}/processing/teste_image{i}.jpg",img)
        #gray_image= grayscale(img)
        thresh, im_bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{dir}/processing/testebw_image{i}.jpg",im_bw)
        noise_removal_image=noise_removal(im_bw)
        deskewed_image=deskew(noise_removal_image)
        cv2.imwrite(f"{dir}/processing/rotate_image{i}.jpg",deskewed_image)
    
    
    def load_and_convert_image(image_path):
        image = cv2.imread(image_path)
        image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        text=image[1750:,:]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=image[0:1750,:]
        gray_image=gray_image[0:1750,:]
        return image, gray_image, text

    def getSkewAngle(cvImage):
        newImage=cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (17, 17), 0)
        thresh = cv2.threshold(blur, 0 , 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        erode = cv2.erode(thresh, kernel , iterations= 2)
        
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        for c in contours:
            rect= cv2.boundingRect(c)
            x, y, w, h=rect
            cv2.rectangle(newImage,(x,y),(x+w,y+h),(36,255,12),2)
            
        largestContour= contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        
        if angle < -45:
            angle = 90 + angle
        elif angle < 45:
            angle = 90 + angle

        return -1.0 * angle

    def rotateImage(cvImage, angle: float):
        (h, w) = cvImage.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Novo tamanho da imagem rotacionada
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        newImage = cv2.warpAffine(cvImage, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    def deskew(cvImage):
        angle = getSkewAngle(cvImage)
        return cv2.rotate(rotateImage(cvImage, -1.0*angle),cv2.ROTATE_90_CLOCKWISE)

    def generate_rois(cvImage):
        newImage = cvImage
        newImage = cv2.resize(newImage, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ordenar contornos primeiro pelo y (topo para baixo) e depois pelo x (esquerda para direita)
        contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

        for i, c in enumerate(contours):
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            x, y, w, h = cv2.boundingRect(box)
            if w > 20:
                roi = newImage[y:y+h, x:x+w]
                cv2.rectangle(newImage, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.imwrite(f"{dir}/processing/roi_text_image{i}.jpg", roi)
                
    def read_rois(roi_list):
        result=[]
        for i, roi in enumerate(roi_list):
            image = Image.open(roi).convert("RGB")

            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            result.append({"ocr": generated_text })
            
        def delete_rois(roi_list):
            for file in roi_list:
                os.remove(file)
        
        delete_rois(roi_list)
        return result

        
    def adjust_string(data):
        cleaned_words = [re.sub(r'[^a-zA-Z0-9]', '', item['ocr']).lower() for item in data]
        return "_".join(filter(None, cleaned_words))

    image_list=glob.glob(f"{dir}/processing/rotate_image*.jpg")

    page_names=[]
    for i, img in enumerate(image_list):
        image, gray_image, text = load_and_convert_image(img)
        deskewed_image = deskew(text)
        generate_rois(deskewed_image)
        roi_list = glob.glob(f"{dir}/processing/roi_text_image*.jpg")
        result=read_rois(roi_list)
        page_names.append(adjust_string(result))
        
    image_list=glob.glob(f"{dir}/processing/rotate_image*.jpg")

    def detect_edges(gray_image, threshold1=50, threshold2=150):
        edges = cv2.Canny(gray_image, threshold1, threshold2, apertureSize=3)
        return edges

    def detect_lines(edges, threshold=30, min_line_length=40, max_line_gap=15):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

    def merge_nearest_lines(lines, image, threshold=45):
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 3:  # Horizontal line
                horizontal_lines.append(y1)
            elif abs(x1 - x2) < 3:  # Vertical line
                vertical_lines.append(x1)

        horizontal_lines = sorted(set(horizontal_lines))
        vertical_lines = sorted(set(vertical_lines))

        def merge_lines(line_positions, threshold):
            merged_lines = []
            current_line = line_positions[0]

            for line in line_positions[0:]:
                if line - current_line <= threshold:
                    continue
                else:
                    merged_lines.append(current_line)
                    current_line = line

            merged_lines.append(current_line)
            return merged_lines

        merged_horizontal_lines = merge_lines(horizontal_lines, threshold)
        merged_vertical_lines = merge_lines(vertical_lines, threshold)

        image_with_merged_lines = np.copy(image)

        for y in merged_horizontal_lines:
            cv2.line(image_with_merged_lines, (0, y), (image.shape[1], y), (0, 255, 0), 2)

        for x in merged_vertical_lines:
            cv2.line(image_with_merged_lines, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)

        return image_with_merged_lines, merged_horizontal_lines, merged_vertical_lines

    def index_and_crop_squares(image, horizontal_lines, vertical_lines):
        index = 1
        horizontal_lines_excluding_top = horizontal_lines[0:]
        cropped_squares = []

        for i in range(len(horizontal_lines_excluding_top) - 1):
            for j in range(len(vertical_lines) - 1):
                top_left = (vertical_lines[j], horizontal_lines_excluding_top[i])
                bottom_right = (vertical_lines[j + 1], horizontal_lines_excluding_top[i + 1])
                
                # Crop the square
                cropped_square = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                cropped_squares.append((index, cropped_square))
                index += 1
        return cropped_squares, index-1

    def analyze_squares(image, num_squares,cropped_squares):
        saved_squares = []
        for i in range(num_squares):
            square_index, square_image = cropped_squares[i]
            square_path = f'{dir}/processing_squares/square_{square_index}.jpg'
            cv2.imwrite(square_path, square_image)
            saved_squares.append(square_path)
        
        return saved_squares

    def extract_number(filename):
        match = re.search(r'square_(\d+)', filename)  # Busca "square_X" e extrai X
        return int(match.group(1)) if match else float('inf')

    def detect_x_in_images(image_folder,num_cols,start_row=1,start_col=1,buffer_ratio=0.3, threshold_black=0.2):
        
        results = []
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=extract_number  # Ordenação correta para 'square_X'
        )
        
        for idx, image_file in enumerate(image_files,start=1):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Converte para escala de cinza
            
            h, w = image.shape
            buffer_size = int(min(h, w) * buffer_ratio)  # Define o tamanho do buffer
            
            # Define a região central onde o X será detectado
            x_start, x_end = w // 2 - buffer_size, w // 2 + buffer_size
            y_start, y_end = h // 2 - buffer_size, h // 2 + buffer_size
            
            center_region = image[y_start:y_end, x_start:x_end]
            
            # Aplica threshold para binarizar a imagem (0 = preto, 255 = branco)
            _, binary_image = cv2.threshold(center_region, 128, 255, cv2.THRESH_BINARY)
            
            # Conta a proporção de pixels pretos na região central
            black_pixel_ratio = np.sum(binary_image == 0) / (center_region.size)
            
            # Classifica como "X" se a proporção de pixels pretos for maior que o threshold definido
            has_x = 1 if black_pixel_ratio > threshold_black else 0
            
            # Indexes
            row = start_row + (idx // num_cols)  
            col = start_col + (idx % num_cols) -1 

            
            results.append({"index": idx,"row": row, "col": col, "has_x": has_x})
            os.remove(image_path)
        
        df_results = pd.DataFrame(results).sort_values(by=["row", "col"]).reset_index(drop=True)
        
        return pd.DataFrame(results)

    image_list=glob.glob(f"{dir}/processing/rotate_image*.jpg")

    for i, img in enumerate(image_list):
        if not os.path.exists(f'{dir}/processing_squares'):
            os.makedirs(f'{dir}/processing_squares')
        
        image, gray_image, text=load_and_convert_image(img)
        edges = detect_edges(gray_image)
        lines = detect_lines(edges)
        
        _, merged_horizontal_lines, merged_vertical_lines = merge_nearest_lines(lines, image)
        cropped_squares, index = index_and_crop_squares(image, merged_horizontal_lines, merged_vertical_lines)
        saved_square_paths=analyze_squares(image,index,cropped_squares)
        image_folder = f"{dir}/processing_squares/"
        df_results = detect_x_in_images(image_folder,num_cols=50)
        df_results.to_csv(f"{dir}/{page_names[i]}.csv", index=False)


    
    
# interface
root = tk.Tk()
root.title("MOCK Processor")
root.geometry("500x300")  # Define o tamanho da janela (largura x altura)

# selecionar arquivo
btn_selecionar = tk.Button(root, text="Select File", command=selecionar_arquivo)
btn_selecionar.pack(pady=10)

# mostrar arquivo selecionado
label_arquivo = tk.Label(root, text="Nenhum arquivo selecionado", wraplength=300)
label_arquivo.pack()

# rodar análise
btn_rodar = tk.Button(root, text="Process file", command=rodar_analise)
btn_rodar.pack(pady=10)

# loop da interface
root.mainloop()
