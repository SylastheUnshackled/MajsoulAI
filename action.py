# -*- coding: utf-8 -*-
# 获取屏幕信息，并通过视觉方法标定手牌与按钮位置，仿真鼠标点击操作输出
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll

from .classifier import Classify
from ..sdk import Operation

DEBUG = False               # 是否显示检测中间结果


def PosTransfer(pos, M: np.ndarray) -> np.ndarray:
    assert(len(pos) == 2)
    return cv2.perspectiveTransform(np.float32([[pos]]), M)[0][0]


def Similarity(img1: np.ndarray, img2: np.ndarray):
    assert(len(img1.shape) == len(img2.shape) == 3)
    if img1.shape[0] < img2.shape[0]:
        img1, img2 = img2, img1
    n, m, c = img2.shape
    img1 = cv2.resize(img1, (m, n))
    if DEBUG:
        cv2.imshow('diff', np.uint8(np.abs(np.float32(img1)-np.float32(img2))))
        cv2.waitKey(1)
    ksize = max(1, min(n, m)//50)
    img1 = cv2.blur(img1, ksize=(ksize, ksize))
    img2 = cv2.blur(img2, ksize=(ksize, ksize))
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    if DEBUG:
        cv2.imshow('bit', np.uint8((np.abs(img1-img2) < 30).sum(2) == 3)*255)
        cv2.waitKey(1)
    return ((np.abs(img1-img2) < 30).sum(2) == 3).sum()/(n*m)


def ObjectLocalization(objImg: np.ndarray, targetImg: np.ndarray) -> np.ndarray:
    """
    https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    Feature based object detection
    return: Homography matrix M (objImg->targetImg), if not found return None
    """
    img1 = objImg
    img2 = targetImg
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for i, j in enumerate(matches):
        if len(j) == 2:
            m, n = j
            if m.distance < 0.7*n.distance:
                good.append(m)
                matchesMask[i] = [1, 0]
    print('  Number of good matches:', len(good))
    if DEBUG:
        # draw
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, matches, None, **draw_params)
        img3 = cv2.pyrDown(img3)
        cv2.imshow('ORB match', img3)
        cv2.waitKey(1)
    # Homography
    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if DEBUG:
            # draw
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, (0, 0, 255), 10, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                                   good, None, **draw_params)
            img3 = cv2.pyrDown(img3)
            cv2.imshow('Homography match', img3)
            cv2.waitKey(1)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = None
    assert(type(M) == type(None) or (
        type(M) == np.ndarray and M.shape == (3, 3)))
    return M


def getHomographyMatrix(img1, img2, threshold=0.0):
    # if similarity>threshold return M
    # else return None
    M = ObjectLocalization(img1, img2)
    if type(M) != type(None):
        print('  Homography Matrix:', M)
        n, m, c = img1.shape
        x0, y0 = np.int32(PosTransfer([0, 0], M))
        x1, y1 = np.int32(PosTransfer([m, n], M))
        sub_img = img2[y0:y1, x0:x1, :]
        S = Similarity(img1, sub_img)
        print('Similarity:', S)
        if S > threshold:
            return M
    return None


class Layout:
    size = (1920, 1080)                                     # 界面长宽
    duanWeiChang = (1348, 321)                              # 段位场按钮
    menuButtons = [(1382, 406), (1382, 573), (1382, 740),
                   (1383, 885), (1393, 813)]   # 铜/银/金之间按钮
    tileSize = (95, 152)                                     # 自己牌的大小


class GUIInterface:

    def __init__(self):
        self.M = None  # Homography matrix from (1920,1080) to real window
        # load template imgs
        join = os.path.join
        root = os.path.dirname(__file__)
        def load(name): return cv2.imread(join(root, 'template', name))
        self.menuImg = load('menu.png')         # 初始菜单界面
        if (type(self.menuImg)==type(None)):
            raise FileNotFoundError("menu.png not found, please check the Chinese path")
        assert(self.menuImg.shape == (1080, 1920, 3))
        self.chiImg = load('chi.png')
        self.pengImg = load('peng.png')
        self.gangImg = load('gang.png')
        self.huImg = load('hu.png')
        self.zimoImg = load('zimo.png')
        self.tiaoguoImg = load('tiaoguo.png')
        self.liqiImg = load('liqi.png')
        self.liujuImg = load('liuju.png')
        # load classify model
        self.classify = Classify()
        self.hwnd = None
        self.find_chrome_window()
        self.last_active = True
        
        # 添加日志文件句柄
        self.log_file = open('mjai_communication.log', 'a', encoding='utf-8')

        # 注册窗口事件监听
        self.old_win_proc = win32gui.SetWindowLong( 
                                                   
            self.hwnd,
            win32con.GWL_WNDPROC,
            self._window_proc
        )

    def _window_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_ACTIVATE:
            is_active = wparam & 0xFFFF != 0
            if not self.last_active and is_active:  # 窗口重新获得焦点
                print("窗口重新激活，等待稳定...")
                time.sleep(1)  # 等待窗口稳定
            self.last_active = is_active
        
        # 调用原来的窗口过程
        return win32gui.CallWindowProc(self.old_win_proc, hwnd, msg, wparam, lparam)

    def find_chrome_window(self):
        """找到雀魂的Chrome窗口"""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "雀魂麻将" in title and "Chrome" in title:
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if not windows:
            raise Exception("找不到雀魂窗口")
        
        # 如果找到多个窗口，使用第一个
        self.hwnd = windows[0]
        print(f"找到雀魂窗口: {win32gui.GetWindowText(self.hwnd)}")
    
    def screenShot(self):
        """捕获指定窗口的画面"""
        if not self.hwnd:
            return None
        
        # 获取窗口大小
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bot - top
        
        # 初始化所有句柄为 None
        hwndDC = mfcDC = saveDC = saveBitMap = None
        img = None
        
        try:
            # 创建设备上下文
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 创建位图对象
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # 复制窗口内容到位图
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 3)
            
            # 转换为numpy数组
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None
        
        finally:
            # 确保所有资源都被释放
            if saveBitMap:
                win32gui.DeleteObject(saveBitMap.GetHandle())
            if saveDC:
                saveDC.DeleteDC()
            if mfcDC:
                mfcDC.DeleteDC()
            if hwndDC:
                win32gui.ReleaseDC(self.hwnd, hwndDC)
        
        return img.copy() if img is not None else None

    def forceTiaoGuo(self):
        # 如果跳过按钮在屏幕上则强制点跳过，否则NoEffect
        self.clickButton(self.tiaoguoImg, similarityThreshold=0.4 , max_attempts=1)

    def _click(self, x: int, y: int, duration: float = 0.2):
        """使用win32api实现后台点击"""
        # 将相对坐标转换为窗口客户区坐标
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        # 获取窗口客户区的位置
        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.hwnd)
        border_width = ((right - left) - (client_right - client_left)) // 2
        title_height = (bottom - top) - (client_bottom - client_top) - border_width
        
        # 计算实际点击位置
        click_x = left + border_width + x
        click_y = top + title_height + y
        
        # 发送鼠标消息
        lParam = win32api.MAKELONG(x, y)
        win32gui.SendMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
        #time.sleep(duration)
        win32gui.SendMessage(self.hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, lParam)

    def actionDiscardTile(self, tile: str):
        for _ in range(20):
            L = self._getHandTiles()
            for t, (x, y) in reversed(L):
                if t == tile:
                    # 实践证明连点20次极为鲁棒，鼠标可以随便甩
                    
                    # 第一次点击选择牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    time.sleep(0.01)
                    # 第二次点击打出牌
                    self._click(x, y)
                    time.sleep(0.01)
                    # 再次点击确认
                    self._click(x, y)
                    
                    time.sleep(1)
                    # 将鼠标移动到waitPos
                    if hasattr(self, 'waitPos'):
                        lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
                        win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
                    
                    return True
            time.sleep(0.5)
        return False

    def actionChiPengGang(self, type_: Operation, tiles: List[str]):
        # 将鼠标移动到waitPos
        if hasattr(self, 'waitPos'):
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        if type_ == Operation.NoEffect:
            self.clickButton(self.tiaoguoImg)
        elif type_ == Operation.Chi:
            self.clickButton(self.chiImg , similarityThreshold=0.6)#桌布是深蓝色的，吃是绿色的但是居然有0.5的误判率
        elif type_ == Operation.Peng:
            self.clickButton(self.pengImg , similarityThreshold=0.6)#桌布是深蓝色的，只有碰是蓝色的
        elif type_ in (Operation.MingGang, Operation.JiaGang):
            self.clickButton(self.gangImg)

    def actionLiqi(self):
        # 将鼠标移动到waitPos
        if hasattr(self, 'waitPos'):
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        self.clickButton(self.liqiImg)        

    def actionHu(self, similarityThreshold=0.48):
        print("开始执行 actionHu")  # 调试信息1
        
        # 移动鼠标到等待位置
        if hasattr(self, 'waitPos'):
            print(f"移动鼠标到等待位置: {self.waitPos}")  # 调试信息2
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        
        # 点击和按钮
        print(f"尝试点击和按钮，相似度阈值: {similarityThreshold}")  # 调试信息3
        click_result = self.clickButton(self.huImg, similarityThreshold=similarityThreshold)
        print(f"点击和按钮结果: {click_result}")  # 调试信息4
        
        print("准备点击确认按钮 (1045,863)")  # 调试信息5
        self._click(1045, 863)
        print("点击确认按钮完成")  # 调试信息6

    def actionZimo(self):
        # 将鼠标移动到waitPos
        if hasattr(self, 'waitPos'):
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        self.clickButton(self.zimoImg)
        
    def checkZimo(self):
        # 将鼠标移动到waitPos
        if hasattr(self, 'waitPos'):
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        self.clickButton(self.zimoImg , similarityThreshold=0.7)
        
    def actionRyukyoku(self):
        # 将鼠标移动到waitPos
        if hasattr(self, 'waitPos'):
            lParam = win32api.MAKELONG(self.waitPos[0], self.waitPos[1])
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
        self.clickButton(self.liujuImg)
        self._click(1045,863) # 遇到了流局，但是不知道是哪个click起了作用

    def calibrateMenu(self):
        # if the browser is on the initial menu, set self.M and return to True
        # if not return False
        self.M = getHomographyMatrix(self.menuImg, self.screenShot(), threshold=0.7)
        result = type(self.M) != type(None)
        if result:
            self.waitPos = np.int32(PosTransfer([100, 100], self.M))
        return result

    def _getHandTiles(self) -> List[Tuple[str, Tuple[int, int]]]:
        # return a list of my tiles' position
        result = []
        assert(type(self.M) != type(None))
        
        # 检查窗口状态
        screen_img1 = self.screenShot()
        if screen_img1 is None or screen_img1.shape[0] < 100 or screen_img1.shape[1] < 100:
            print(f"Warning: Invalid window size: {screen_img1.shape if screen_img1 is not None else 'None'}")
            time.sleep(1)  # 等待窗口恢复
            screen_img1 = self.screenShot()
            if screen_img1 is None or screen_img1.shape[0] < 100 or screen_img1.shape[1] < 100:
                print("Error: Window size still invalid")
                return []
            
        time.sleep(0.5)
        screen_img2 = self.screenShot()
        if screen_img2 is None or screen_img2.shape != screen_img1.shape:
            print("Error: Inconsistent window size between screenshots")
            return []
        
        screen_img = np.minimum(screen_img1, screen_img2)  # 消除高光动画
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([235, 1002], self.M))
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 110
        tileThreshold = np.int32(0.7*(PosTransfer(Layout.tileSize, self.M)-O))
        fail = 0
        maxFail = np.int32(PosTransfer([100, 0], self.M)-O)[0]
        i = 0
        
        # 获取图像尺寸
        height, width = img.shape[:2]
        
        while fail < maxFail:
            x, y = start[0]+i, start[1]
            
            # 添加边界检查
            if x >= width or y >= height:
                break
            
            try:
                if all(img[y, x, :] > colorThreshold):
                    fail = 0
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    
                    # 确保不会越界
                    if x + dx > width or y + dy > height:
                        continue
                    
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = screen_img[y:y+dy, x:x+dx, :]
                        tileStr = self.classify(tile_img)
                        result.append((tileStr, (x+dx//2, y+dy//2)))
                        i = x+dx-start[0]
                else:
                    fail += 1
                i += 1
            except IndexError:
                print(f"Warning: Index out of bounds at x={x}, y={y}, width={width}, height={height}")
                break
            
        return result

    def clickButton(self, buttonImg, similarityThreshold=0.5, max_attempts=10, retry_delay=0.2):
        """
        点击按钮的通用函数
        buttonImg: 按钮模板图像
        similarityThreshold: 相似度阈值
        max_attempts: 最大重试次数
        retry_delay: 重试间隔时间
        """
        for attempt in range(max_attempts):
            # 截取两次屏幕进行对比，避免动画影响
            screen1 = self.screenShot()
            time.sleep(0.1)
            screen2 = self.screenShot()
            if screen1 is None or screen2 is None:
                continue
            
            # 使用两次截图的最小值来消除可能的动画效果
            screen = np.minimum(screen1, screen2)
            
            # 计算缩放比例
            scale_x = screen.shape[1] / Layout.size[0]
            scale_y = screen.shape[0] / Layout.size[1]
            
            # 按比例调整搜索区域
            x0, y0 = np.int32(PosTransfer([595, 557], self.M))
            x1, y1 = np.int32(PosTransfer([1508, 912], self.M))
            
            # 确保坐标在有效范围内
            x0 = max(0, min(x0, screen.shape[1]))
            x1 = max(0, min(x1, screen.shape[1]))
            y0 = max(0, min(y0, screen.shape[0]))
            y1 = max(0, min(y1, screen.shape[0]))
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            img = screen[y0:y1, x0:x1, :]
            
            # 按比例缩放模板图像
            h, w = buttonImg.shape[:2]
            new_w = int(w * scale_x)
            new_h = int(h * scale_y)
            templ = cv2.resize(buttonImg, (new_w, new_h))
            
            if templ.shape[0] > img.shape[0] or templ.shape[1] > img.shape[1]:
                continue
            
            T = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=templ.copy())
            _, _, (x, y), _ = cv2.minMaxLoc(T)
            
            if DEBUG:
                T = np.exp((1-T/T.max())*10)
                T = T/T.max()
                cv2.imshow('T', T)
                cv2.waitKey(0)
            
            dst = img[y:y+new_h, x:x+new_w].copy()
            dst[templ == 0] = 0
            similarity = Similarity(templ, dst)
            
            # 写入日志
            self.log_file.write(f"Button click attempt {attempt + 1}/{max_attempts}, Similarity: {similarity:.3f}, Threshold: {similarityThreshold}\n")
            self.log_file.flush()  # 确保立即写入文件
            
            if similarity >= similarityThreshold:
                self._click(x+x0+new_w//2, y+y0+new_h//2)
                return True
            
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)
            
        return False

    def clickCandidateMeld(self, tiles: List[str]):
        # 有多种不同的吃碰方法，二次点击选择
        assert(len(tiles) == 2)
        # find all combination tiles
        result = []
        assert(type(self.M) != type(None))
        screen_img = self.screenShot()
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([960, 753], self.M))
        leftBound = rightBound = start[0]
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 200
        tileThreshold = np.int32(0.7*(PosTransfer((78, 106), self.M)-O))
        maxFail = np.int32(PosTransfer([60, 0], self.M)-O)[0]
        for offset in [-1, 1]:
            i = 0
            while True:
                x, y = start[0]+i*offset, start[1]
                if offset == -1 and x < leftBound-maxFail:
                    break
                if offset == 1 and x > rightBound+maxFail:
                    break
                if all(img[y, x, :] > colorThreshold):
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = screen_img[y:y+dy, x:x+dx, :]
                        tileStr = self.classify(tile_img)
                        result.append((tileStr, (x+dx//2, y+dy//2)))
                        leftBound = min(leftBound, x)
                        rightBound = max(rightBound, x+dx)
                i += 1
        result = sorted(result, key=lambda x: x[1][0])
        if len(result) == 0:
            return True  # 其他人先抢先Meld了！
        print('clickCandidateMeld tiles:', result)
        assert(len(result) % 2 == 0)
        for i in range(0, len(result), 2):
            x, y = result[i][1]
            if tuple(sorted([result[i][0], result[i+1][0]])) == tiles:
                self._click(x, y)
                time.sleep(1)
                return True
        raise Exception('combination not found, tiles:',
                        tiles, ' combination:', result)
        return False

    def actionReturnToMenu(self):
        # 在终局以后点击确定跳转回���单主界面
        x, y = np.int32(PosTransfer((1785, 1003), self.M))  # 终局确认按钮
        while True:
            time.sleep(5)
            x0, y0 = np.int32(PosTransfer([0, 0], self.M))
            x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
            img = self.screenShot()
            S = Similarity(self.menuImg, img[y0:y1, x0:x1, :])
            if S > 0.5:
                return True
            else:
                print('Similarity:', S)
                self._click(x, y)

    def actionBeginGame(self, level: int):
        # 从开始界面点击匹配对局, level=0~4 (铜/银/金/王座之间)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.duanWeiChang, self.M))
        self._click(x, y)
        time.sleep(2)
        if level == 4:
            # 王座之间在屏幕外面需要先拖一下
            x1, y1 = np.int32(PosTransfer(Layout.menuButtons[2], self.M))
            x2, y2 = np.int32(PosTransfer(Layout.menuButtons[0], self.M))
            # 模拟拖动
            lParam1 = win32api.MAKELONG(x1, y1)
            lParam2 = win32api.MAKELONG(x2, y2)
            win32gui.SendMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam1)
            time.sleep(0.5)
            win32gui.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, lParam2)
            time.sleep(0.5)
            win32gui.SendMessage(self.hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, lParam2)
            time.sleep(1.5)
        x, y = np.int32(PosTransfer(Layout.menuButtons[level], self.M))
        self._click(x, y)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.menuButtons[1], self.M))  # 四人南
        self._click(x, y)

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            # 关闭日志文件
            if hasattr(self, 'log_file'):
                self.log_file.close()
            
            # 恢复原来的窗口过程
            if hasattr(self, 'old_win_proc') and self.old_win_proc and self.hwnd:
                win32gui.SetWindowLong(
                    self.hwnd,
                    win32con.GWL_WNDPROC,
                    self.old_win_proc
                )
        except:
            pass
