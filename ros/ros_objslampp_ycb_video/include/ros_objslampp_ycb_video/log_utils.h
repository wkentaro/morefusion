#define ANSI_COLOR_RED     "\x1b[31;1m"
#define ANSI_COLOR_GREEN   "\x1b[32;1m"
#define ANSI_COLOR_YELLOW  "\x1b[33;1m"
#define ANSI_COLOR_BLUE    "\x1b[34;1m"
#define ANSI_COLOR_MAGENTA "\x1b[35;1m"
#define ANSI_COLOR_CYAN    "\x1b[36;1m"
#define ANSI_COLOR_WHITE   "\x1b[37;1m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define ROS_INFO_RED(text) ROS_INFO("%s%s%s", ANSI_COLOR_RED, text, ANSI_COLOR_RESET);
#define ROS_INFO_GREEN(text) ROS_INFO("%s%s%s", ANSI_COLOR_GREEN, text, ANSI_COLOR_RESET);
#define ROS_INFO_YELLOW(text) ROS_INFO("%s%s%s", ANSI_COLOR_YELLOW, text, ANSI_COLOR_RESET);
#define ROS_INFO_BLUE(text) ROS_INFO("%s%s%s", ANSI_COLOR_BLUE, text, ANSI_COLOR_RESET);
#define ROS_INFO_MAGENTA(text) ROS_INFO("%s%s%s", ANSI_COLOR_MAGENTA, text, ANSI_COLOR_RESET);
#define ROS_INFO_CYAN(text) ROS_INFO("%s%s%s", ANSI_COLOR_CYAN, text, ANSI_COLOR_RESET);
#define ROS_INFO_WHITE(text) ROS_INFO("%s%s%s", ANSI_COLOR_WHITE, text, ANSI_COLOR_RESET);
