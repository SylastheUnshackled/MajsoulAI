使用方法：

更新majsoul_wrapper

雀魂登录界面F12获取liqi.json，覆盖

在proto目录下执行：

npm install -g protobufjs-cli

pbjs -t proto3 liqi.json > liqi.proto

protoc --python_out=. liqi.proto

注意protoc版本和protobuf版本一致，具体问copilot

在MajsoulAI目录下执行：

python -m majsoul_wrapper

在打开的浏览器中打开雀魂

然后使用remote.py启动WSL内的Akochan

然后回到MajsoulAI目录，执行：

python main.py --remote_ip WSL_IP

露出雀魂界面后，AI会校准后等待游戏开始。

-d 可以指定预期打多少分钟，时限40分钟前会不开新局，也可以通过删除canloop.flag文件来强制不开新局

可以截下自己的主屏幕，调整到1920*1080替换majsoul_wrapper\action\template\menu.png，来提高主界面的识别率

相较于https://github.com/747929791/majsoul_wrapper.git，将main.py改为了main2.py，修改了action.py和sdk.py来适配mjailog，~~并且测试了可以正常对接akochan而不需修改akochan。使用只需将remote.py放到akochan的system.exe同级目录，启动remote.py即可。~~

对接MORTAL


如果打开mitm后没有出现源源不断的数据交换流输出log，那么很可能是之前的mitm或者python进程没关，目前我只有一次复现，通过两次重启电脑（？）解决了这个问题
（吃碰杠）图标使用的是繁体中文，和简体中文有细微的差别，如果不使用繁体中文可能会有小bug。（使用也可能会有，~~流局目前没测试过~~已测试）
