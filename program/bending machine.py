
# bending machine

coffee=10
while True:
    money=int(input("돈을 넣어주세요:"))
    if money==300:
        print("커피가 나왔습니다.")
        coffee=coffee-1
        print("남은 커피는 %s잔입니다."%coffee)
    elif money>300:
        print("커피가 나왔습니다. 거스름돈은 %s원 입니다."%(money-300))
        coffee=coffee-1
        print("남은 커피는 %s잔입니다."%coffee)
    else:
        print("돈이 %s원 부족합니다."%(300-money))
    while money<300:
        money1=int(input("돈을 %s원 더 넣어주세요:"%(300-money)))
        money=money+money1
        if money==300:
            print("커피가 나왔습니다.")
            coffee=coffee-1
            print("남은 커피는 %s잔입니다."%coffee)
        elif money>300:
            print("커피가 나왔습니다. 거스름돈은 %s원 입니다."%(money-300))
            coffee=coffee-1
            print("남은 커피는 %s잔입니다."%coffee)
    if not coffee:
        print("Sold out")
        break
