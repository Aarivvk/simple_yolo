#!/usr/bin/python3

# This script can be used for building the project and other project related tasks.

import os
import sys

build_cmd = "./scripts/build.sh"
bulid_status=0

MAIN_OPS = {"build": "builds the packages for you",
            "clean": "Clean the install build and log folder",
            "run": "Runs the application"}
BUILD_OPS = {"all":"builds all the packages",
             "package": "builds selected packages"}

def print_oops(ops):
    for key, value in ops.items():
        print(key + " : " + value)


def build(sub_cmd, packages):
    if (sub_cmd == list(BUILD_OPS.keys())[0]):
        bulid_status = os.system(build_cmd)

    elif (sub_cmd == list(BUILD_OPS.keys())[1]):
        bulid_status = -1
        print("Not supported yet!")
        #TODO: Add pc and nandhi case.
    else:
        print("Invalid command\noptions are:")
        print_oops(BUILD_OPS)


def run_option():
    if len(sys.argv) == 2:
        print("options are:")
        print_oops(MAIN_OPS)
        exit()

    cmd = sys.argv[1]
    if cmd == list(MAIN_OPS.keys())[0]:
        param = ""
        sub_cmd = sys.argv[2]
        for i in range(3, len(sys.argv)):
            param = param + " " + sys.argv[i] + " "
        build(sub_cmd, param)
    elif cmd == list(MAIN_OPS.keys())[1]:
        bulid_status = os.system("rm -rf build install log")
        print("Deleted the build artifacts")
    elif cmd == list(MAIN_OPS.keys())[2]:
        os.system("./scripts/run.sh")
    else:
        print("Invalid command\noptions are:")
        print_oops(MAIN_OPS)


def main():
    print("Ambhaaa!")

    run_option()
    print("The build return vaule " + str(bulid_status))
    exit(bulid_status)


if __name__ == "__main__":
    main()
