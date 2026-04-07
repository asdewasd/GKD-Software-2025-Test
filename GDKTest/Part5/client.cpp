#include <iostream>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr));

    // 构造输入
    std::string msg = "1 784 ";
    for (int i = 0; i < 784; i++) {
        msg += "0.5 ";
    }

    send(sock, msg.c_str(), msg.size(), 0);

    char buffer[4096] = {0};
    recv(sock, buffer, sizeof(buffer), 0);

    std::cout << "📤 返回结果:\n" << buffer << std::endl;

    closesocket(sock);
    WSACleanup();
}