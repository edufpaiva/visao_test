'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - link rotacao: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    - python -m pip install --upgrade pip 
    - python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
    - pip install opencv-python
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
from os import system
system("cls")

import cv2
import numpy as np
import math

PATH_BASE = "./bases/"
PATH_DEF = "./defeitos/"
N_IMAGENS_BASE = 1
N_IMAGENS_DEF = 2
INDEX_BASE = 0
INDEX_DEF = 1
RESIZE = 35
RESIZE = 100


class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        return("x: %i, y: %i" %(self.x, self.y))

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
    h_1, w_1 = img1.shape[:2]
    h_2, w_2 = img2.shape[:2]
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
    # show_resized_img(result, RESIZE)
    return result
    
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

def get_ponto_superior(img):
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if img[y][x] != 255:
                return Ponto(x, y)

    return Ponto(0, 0)

def get_pontos_linha_superior(img, ponto, step=10):
    h, w = img.shape

    pontos = []

    if ponto.x < w/2:
        for x in range(ponto.x, w, step):
            for y in range(int(h/4)):
                if img[y][x] != 255:
                    pontos.append(Ponto(x,y))
                    break
    else:
        for x in range(w, ponto.x, -1 *step):
            for y in range(int(h/4)):
                try:
                    pass
                    if img[y][x] != 255:
                        pontos.append(Ponto(x,y))
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
    if p1.x-p2.x == 0: return 0
    return math.degrees(math.atan((p1.y-p2.y)/(p1.x-p2.x)))
    
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

def format_img(img):
    ponto_mais_alto = get_ponto_superior(img)
    ponto_mais_a_esqueda = get_ponto_esquerda(img)    
    pontos_superiores = get_pontos_linha_superior(img, ponto_mais_alto, 50)
    angulos = []
    # print("ponto mais alto", ponto_mais_alto.to_string())
    # print(pontos_superiores)
    for ponto in pontos_superiores:
        angulos.append(get_angulo(ponto_mais_alto, ponto))
    # print(angulos)
    moda = get_moda(angulos)
    print("moda:", moda)
    img = rotate_bound(img, -moda)
    img = retira_bordas(img)

    ponto_mais_alto_rotate = get_ponto_superior(img)
    ponto_mais_a_esqueda_rotate = get_ponto_esquerda(img)    
    
    
    img = ajusta_img(img, ponto_mais_a_esqueda_rotate.x - ponto_mais_a_esqueda.x, ponto_mais_alto_rotate.y - ponto_mais_alto.y)
    img = retira_bordas(img)
    show_img(img)
    
    print(ponto_mais_alto.to_string(), ponto_mais_alto_rotate.to_string())
    
    return img

def ajusta_img(img, off_x = 0, off_y = 0):
    result = get_empty_img_grayscale(img)
    h, w = img.shape[:2]

    for y in range(off_y, h-off_y):
        for x in range(off_x, w-off_x):
            result[y-off_y][x-off_x] = img[y][x]
    return result

def retira_bordas(img):
    h, w = img.shape[:2]

    print(w,h)
    for y in range(h-1, 0, -1):
        for x in range(w-1, 0, -1):
            if img[y][x] == 255: break
            img[y][x] = 255

    for y in range(h):
        for x in range(w):
            if img[y][x] == 255: break
            img[y][x] = 255

    for x in range(w-1, 0, -1):
        for y in range(h-1, 0, -1):
            if img[y][x] == 255: break
            img[y][x] = 255
    
    for x in range(w):
        for y in range(h):
            if img[y][x] == 255: break
            img[y][x] = 255

    
    return img

def get_moda(vet):
    vet_len = len(vet)
    if vet_len%2 == 0:
        vet_len = int(vet_len/2)
        return (vet[vet_len] + vet[vet_len-1])/2
    else:
        return vet[int(vet_len/2)]

def get_media(vet):
    total = 0
    for i in vet: total += i
    return total / len(vet)

def get_ponto_esquerda(img):
    h, w = img.shape[:2]

    for x in range(w):
        for y in range(h):
            if img[y][x] != 255:
                return Ponto(x,y)
    return Ponto(0,0)

bases =  get_img_base(PATH_BASE)
defeitos = get_img_def(PATH_DEF)


img1 = percorre_pixels(bases[INDEX_BASE])
img2 = percorre_pixels(defeitos[INDEX_DEF])

# img2 = format_img(img2)




result = compare_img(img1, img2)
# show_img(result)

img2 = format_img(img2)
# show_img(img2)

result = compare_img(img1, img2)
show_img(result)


# ponto = verifica_pixel_mais_alto(img2)
# result = get_empty_img(img2)


# pontos = get_pontos_linha_superior(img2, ponto, int(img2.shape[1]/10))
# pontos = get_pontos_linha_superior(img2, ponto, 50)
# media = 0
# for i in pontos:
#     if i == ponto:
#         continue
#     result = print_ponto(result, i)
#     result = print_ponto(result, {"x": i["x"], "y":ponto["y"]}, [255, 0, 0])
#     angulo = get_angulo(ponto, i)
#     if angulo < 1 and angulo > 0:
#         if media == 0:
#             media = angulo
#         else:
#             media = (media + angulo) / 2
#         print(angulo)
#     else:
#         pass
# print("angulo medio", media)

# result = print_ponto(result, ponto, [0,255,0])

# show_resized_img(rotate_bound(img2, -media), RESIZE)
# img2 = rotate_bound(img2, -media)

# compare_img(img1, img2)

# print(ponto)
# show_resized_img(result, RESIZE)
# show_resized_img(bases[INDEX_BASE], RESIZE)


# print()
# # print(verifica_pixel_mais_alto(imagens[IMG_2_INDEX]))


# print(ponto, pontos[len(pontos)-2])




