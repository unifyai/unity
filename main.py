import unify

from user_requests import make_request

unify.activate("Unity")

make_request()
while True:
    input("press enter to make another request, press ctrl-C to exit\n")
    make_request()
