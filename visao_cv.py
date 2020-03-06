import cv2

def read_and_show_img(path):
    #Le uma imagem
    img = cv2.imread(path)
    #AlunosPUC_Oracle.jpg')#, cv2.IMREAD_GRAYSCALE )
    #mostra na tela
    cv2.imshow('image', img)
    #espera pressionar uma tecla
    cv2.waitKey(0)

read_and_show_img("base1.jpg")

