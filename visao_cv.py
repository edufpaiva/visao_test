'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - link rotacao: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    - python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
from os import system
system("cls")

import cv2
import numpy as np
import math

PATH_BASE = "./bases/02 - parte/"
PATH_DEF = "./defeitos/02 - parte/"
N_IMAGENS_BASE = 1
N_IMAGENS_DEF = 4
INDEX_BASE = 0
INDEX_DEF = 3
RESIZE = 35
RESIZE = 80


class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def merge_sort(vet):
    if len(vet) == 1:
        return vet
    mid = int(len(vet)/2)
    return merge(merge_sort(vet[:mid]), merge_sort(vet[mid:]))


    print(vet[:mid])
    print(vet[mid:])

def merge(vet_1, vet_2):
    i = j = 0
    ret = []
    while i < len(vet_1) and j < len(vet_2):
        pass
        if vet_1[i] < vet_2[j]:
            ret.append(vet_1[i])
            i += 1
        else:
            ret.append(vet_2[j])
            j += 1

    for k in range(i, len(vet_1)):
        ret.append(vet_1[k])
    for k in range(j, len(vet_2)):
        ret.append(vet_2[k])
    
    return ret

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

def get_img_base(path = "./"):
    imagens = []
    for i in range(1,N_IMAGENS_BASE+1):
        imagens.append(read_img( path +"base (%s).jpg" % i))
    return imagens

def get_img_def(path = "./"):
    imagens = []
    for i in range(1,N_IMAGENS_DEF + 1):
        imagens.append(read_img(path +"def (%s).jpg" % i))
    return imagens


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
            try:
                if img1[py][px] == 255 and img2[py][px] == 255: continue
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
                try:
                    pass
                    if img[y][x] != 255:
                        pontos.append({"x": x, "y":y})
                        break
                except :

                    pass
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
    
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# read_and_show_img("./defeitos/02 - parte/def (2).jpg")


# mostra_imagens_base()

bases =  get_img_base(PATH_BASE)
defeitos = get_img_def()


# show_resized_img(imagens[7], 20)

bases[INDEX_BASE] = percorre_pixels(bases[INDEX_BASE])
defeitos[INDEX_DEF] = percorre_pixels(defeitos[INDEX_DEF])

img1 = bases[INDEX_BASE]
img2 = defeitos[INDEX_DEF]

# show_resized_img(img1, RESIZE)
# show_resized_img(img2, RESIZE)

compare_img(img1, img2)

ponto = verifica_pixel_mais_alto(img2)
result = get_empty_img(img2)


pontos = get_pontos_linha_superior(img2, ponto, int(img2.shape[1]/10))
pontos = get_pontos_linha_superior(img2, ponto, 50)
media = 0
for i in pontos:
    if i == ponto:
        continue
    result = print_ponto(result, i)
    result = print_ponto(result, {"x": i["x"], "y":ponto["y"]}, [255, 0, 0])
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

show_resized_img(rotate_bound(img2, -media), RESIZE)
img2 = rotate_bound(img2, -media)

compare_img(img1, img2)

print(ponto)
show_resized_img(result, RESIZE)
show_resized_img(bases[INDEX_BASE], RESIZE)


print()
# print(verifica_pixel_mais_alto(imagens[IMG_2_INDEX]))


print(ponto, pontos[len(pontos)-2])




