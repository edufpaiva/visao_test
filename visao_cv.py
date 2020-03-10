'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
from os import system
system("cls")

import cv2
import numpy as np
import math

N_IMAGENS = 11
IMG_1_INDEX = 0
IMG_2_INDEX = 5
RESIZE = 80


#   Le uma imagem
def read_img(path):
    #AlunosPUC_Oracle.jpg')#, cv2.IMREAD_GRAYSCALE )
    # Le a imagem para uma matriz
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

#   Mostra img na tela
def show_resized_img(img, percent):
    dim = img.shape
    print(img.shape)
    
    dim = (int(dim[0] * percent / 100), int(dim[1] * percent / 100))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)

#   Mostra img redimensionada na tela
def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)



#   Print em console: widht, height and channels of img
def print_img_params(img):
    print(img.shape)
    print ("width: {} pixels".format(img.shape[1]))
    print ("height: {} pixels".format(img.shape[0]))
    if len(img.shape) >= 3: print ("channels: {}".format(img.shape[2])) #   bites por pixel


def read_and_show_img(path):
    print("img: ", path)
    img = read_img(path)
    show_img(img)
    print_img_params(img)
    print()

def get_img_base():
    imagens = []
    for i in range(1,N_IMAGENS+1):
        imagens.append(read_img("base%s.jpg" % i))
    return imagens

def mostra_imagens_base():
    #   mostra todas a imagens contidas na pasta
    #   que se chamam base e sao numeradas de 1 a 6 ;D hehe
    for i in range(1,7):
        read_and_show_img("base%s.jpg" % i)

#   Media de pixels da imagem -  para imagens BGR 
def media(vet):
    total = 0
    for i in vet:
        total += i
    return int(total/len(vet))

def percorre_pixels(img):
    # print(img.shape)
    print_img_params(img)
    # height, width, channels = img.shape # comment if use in grayscale
    height, width = img.shape
    
    for py in range(0, height):
        for px in range(0,  width):
            # pixel = int(media(img[py][px]) / 50) * 50 #   comment is use iin grayscale
            pixel = int(img[py][px] / 50) * 50
            if pixel >= 200: pixel = 255
            if pixel <= 100 : pixel = 0
            img[py][px] = pixel

            # for pixel_index in range(len(img[py][px])): #   comment is use iin grayscale
            #     img[py][px][pixel_index] = pixel

    return img
    # show_img(img)

def get_empty_img(img):
    h_1, w_1 = img.shape
    return np.zeros((h_1, w_1, 3), np.uint8)

def get_empty_img_grayscale(img):
    h_1, w_1 = img.shape
    return np.zeros((h_1, w_1, 1), np.uint8)


def compare_img(img1, img2):
    print("comparando")
    h_1, w_1 = img1.shape
    h_2, w_2 = img2.shape
    print(h_1, h_2, w_1, w_2)


    result = np.zeros((h_1, w_1, 3), np.uint8)


    for py in range(h_1):
        for px in range(w_1):
            if img1[py][px] == 255 and img2[py][px] == 255: continue
            try:
                if compare_pixel(img1, img2, py, px):
                    result[py][px][1] = 255
                    pass
                else: 
                    result[py][px][2] = 255
            except:
                # print(py, px)
                result[py][px][0] = 255
                pass
    # show_img(result)
    show_resized_img(result, RESIZE)
    
def compare_pixel(img1, img2, y, x):
    pixel = img1[y][x]
    
    h_1, w_1 = img1.shape
    
    for i in range(y-1, y+2):
        if i > img2.shape[0]: continue
        for j in range(x, x+2):
            if i >= h_1: continue
            if j >= w_1: continue
            if i < 0 or j < 0: continue
            if img2[i][j] == pixel: return True
    return False

def verifica_pixel_mais_alto(img):
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if img[y][x] != 255:
                return {"x":x, "y":y  }

    return {"x":0, "y":0 }

def get_pontos_linha_superior(img, ponto, step=10):
    h, w = img.shape

    pontos = []

    if ponto["x"] < w/2:
        for x in range(ponto["x"], w, step):
            for y in range(int(h/4)):
                if img[y][x] != 255:
                    pontos.append({"x": x, "y":y})
                    break
    else:
        for x in range(w, ponto["x"], -1 *step):
            for y in range(int(h/4)):
                if img[y][x] != 255:
                    pontos.append({"x": x, "y":y})
                    break
    return pontos

def print_ponto(img, ponto, bgr = [0, 0, 255]):
    # print(img.shape)

    h_1, w_1, channel = img.shape
    x = ponto["x"]
    y = ponto["y"]

    for i in range(y-2, y+4):
        for j in range(x-2, x+4):
            img[i][j] = bgr
    return img
    
def get_angulo(p1, p2):
    if p1["x"]-p2["x"] == 0: return 0
    return math.degrees(math.atan((p1["y"]-p2["y"])/(p1["x"]-p2["x"])))
    






# mostra_imagens_base()

imagens =  get_img_base()



# show_resized_img(imagens[7], 20)

imagens[IMG_1_INDEX] = percorre_pixels(imagens[IMG_1_INDEX])
imagens[IMG_2_INDEX] = percorre_pixels(imagens[IMG_2_INDEX])

# compare_img(imagens[IMG_1_INDEX], imagens[IMG_2_INDEX])

ponto = verifica_pixel_mais_alto(imagens[IMG_1_INDEX])
result = get_empty_img(imagens[IMG_1_INDEX])


pontos = get_pontos_linha_superior(imagens[IMG_1_INDEX], ponto, int(imagens[IMG_1_INDEX].shape[1]/10))
pontos = get_pontos_linha_superior(imagens[IMG_1_INDEX], ponto, 50)
media = 0
for i in pontos:
    if i == ponto:
        continue
    result = print_ponto(result, i)
    # result = print_ponto(result, i, [255, 0, 0])
    angulo = get_angulo(ponto, i)
    if angulo < 1 and angulo > 0:
        if media == 0:
            media = angulo
        else:
            media = (media + angulo) / 2
        print(angulo)
    else:
        pass
print("angulo medio", media)

result = print_ponto(result, ponto, [0,255,0])



print(ponto)
show_resized_img(result, RESIZE)
show_resized_img(imagens[IMG_1_INDEX], RESIZE)


print()
# print(verifica_pixel_mais_alto(imagens[IMG_2_INDEX]))


print(ponto, pontos[len(pontos)-2])

