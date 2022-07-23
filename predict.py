import preprocessing as x

whatisit=x.predictirl('dogimage.jpg')

if whatisit==0:
    print("cat")
else:
    print("dog")
