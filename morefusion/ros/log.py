import termcolor


def loginfo_red(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="red", attrs={"bold": True}))


def loginfo_green(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="green", attrs={"bold": True}))


def loginfo_yellow(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="yellow", attrs={"bold": True}))


def loginfo_blue(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="blue", attrs={"bold": True}))


def loginfo_magenta(msg):
    import rospy

    rospy.loginfo(
        termcolor.colored(msg, color="magenta", attrs={"bold": True})
    )


def loginfo_cyan(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="cyan", attrs={"bold": True}))


def loginfo_white(msg):
    import rospy

    rospy.loginfo(termcolor.colored(msg, color="white", attrs={"bold": True}))
