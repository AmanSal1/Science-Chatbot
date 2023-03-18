from main import response
import random
import tensorflow as tf




if __name__ == '__main__':
    while True:

        query = input(">>>")
        if query in [ "Bye", "See you later", "Goodbye","bye","goodbye"]:
            print(random.choice([ "See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon." ]))
            break


        else:
            response(query)

