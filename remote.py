# -*- coding: utf-8 -*-
import time
import select
import socket
import subprocess
import json
import os

# for remote client
REMOTE_HOST = '0.0.0.0'
REMOTE_PORT = 14782

# 获取mortal目录的绝对路径
MORTAL_DIR = os.path.dirname(os.path.abspath(__file__))

def GameLoop(client_conn):
    mortal_process = None
    while True:
        try:
            # 从客户端接收数据
            data = client_conn.recv(1024)
            if not data:
                print('Client disconnected')
                break
            
            # 解析数据
            try:
                # 将接收到的数据按行分割
                messages = data.decode('utf-8').strip().split('\n')
                for message in messages:
                    event = json.loads(message)
                    # 检测start_game事件并启动进程
                    if mortal_process is None and event.get('type') == 'start_game':
                        player_id = event.get('id', 0)
                        print(f"Starting mortal.py with player_id: {player_id}")
                        
                        # 获取正确的工作目录（Mortal根目录）
                        mortal_root = os.path.dirname(MORTAL_DIR)  # 从 mortal 目录向上一级到 Mortal 目录
                        print(f"Setting working directory to: {mortal_root}")  # 添加调试信息
                        
                        # 设置环境变量
                        env = os.environ.copy()
                        env['MORTAL_REVIEW_MODE'] = '0'
                        
                        # 使用 python 命令运行 mortal.py
                        mortal_process = subprocess.Popen(
                            ["python", "mortal/mortal.py", str(player_id)],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            cwd=mortal_root,  # 使用Mortal根目录作为工作目录
                            env=env  # 传入修改后的环境变量
                        )
                        
                        # 等待mortal.py加载完成
                        print("Waiting for mortal.py to initialize...")
                        # 可以通过检查stderr来确认加载完成
                        while True:
                            line = mortal_process.stderr.readline()
                            print(f"mortal.py stderr: {line}", flush=True)  # 添加这行来输出所有错误信息
                            
                            if "loaded state" in line:
                                print("Mortal.py is ready")
                                break
                            # 添加超时检查
                            if mortal_process.poll() is not None:
                                # 获取所有剩余的错误输出
                                remaining_errors = mortal_process.stderr.readlines()
                                print(f"Additional errors: {remaining_errors}", flush=True)
                                raise Exception(f"Mortal.py failed to start. Exit code: {mortal_process.returncode}")
            except json.JSONDecodeError:
                print("Invalid JSON received")
                print(f"Raw data: {data.decode('utf-8')}", flush=True)  # 添加这行来输出原始数据
                continue
            
            # 如果进程还没启动，跳过
            if mortal_process is None:
                print("Waiting for start_game event...")
                continue
                
            # 发送到mortal.py的stdin
            print(f"Sending to mortal.py: {data.decode('utf-8')}", flush=True)
            try:
                mortal_process.stdin.write(data.decode('utf-8') + '\n')
                mortal_process.stdin.flush()
            except IOError as e:
                print(f"Failed to write to mortal.py: {e}")
                # 检查进程状态
                if mortal_process.poll() is not None:
                    print(f"Mortal.py has exited with code: {mortal_process.returncode}")
                    remaining_errors = mortal_process.stderr.readlines()
                    print(f"Final errors from mortal.py: {remaining_errors}")
                break
            
            # 等待响应，最多等待10次
            for _ in range(10):
                try:
                    readable, _, _ = select.select([mortal_process.stdout, mortal_process.stderr], [], [], 0.1)
                    
                    got_response = False
                    for fd in readable:
                        if fd == mortal_process.stdout:
                            response = fd.readline()
                            if not response:
                                print('Mortal.py stdout closed')
                                continue
                            print(f"Received from mortal.py stdout: {response}", flush=True)
                            # 发送回客户端
                            client_conn.send(response.encode('utf-8'))
                            got_response = True
                        elif fd == mortal_process.stderr:
                            error = fd.readline()
                            if error:
                                print(f"mortal.py stderr: {error}", flush=True)
                    
                    if got_response:
                        break
                        
                except Exception as e:
                    print(f"Error during select/read: {e}")
                    break
            
            if mortal_process.poll() is not None:
                print(f"Mortal.py exited with code: {mortal_process.returncode}")
                break
                
        except Exception as e:
            print(f'Error in GameLoop: {e}')
            break
    
    # 清理
    if mortal_process:
        try:
            mortal_process.terminate()
            mortal_process.wait(timeout=5)  # 等待进程结束
        except:
            mortal_process.kill()  # 强制结束
        finally:
            mortal_process.stdout.close()
            mortal_process.stderr.close()
    client_conn.close()

if __name__ == '__main__':
    while True:
        try:
            remote_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            remote_server_address = (REMOTE_HOST, REMOTE_PORT)
            remote_server.bind(remote_server_address)
            remote_server.listen(1)

            print('Remote server starting up on %s port %s' % remote_server_address)
            
            while True:
                print('Waiting for client.')
                client_conn, client_address = remote_server.accept()
                try:
                    print('Client connection: ', client_address)
                    client_conn.send(b'ACK')
                    print('Server is ready.')
                    GameLoop(client_conn)
                except Exception as e:
                    print(f'Error handling client: {e}')
                finally:
                    client_conn.close()
        except Exception as e:
            print(f'Server error: {e}')
        finally:
            remote_server.close()