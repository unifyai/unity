from user_requests import make_request

input("press enter to make a request, press ctrl-C to exit\n")
make_request()
while True:
    input("press enter to make another request, press ctrl-C to exit\n")
    make_request()
