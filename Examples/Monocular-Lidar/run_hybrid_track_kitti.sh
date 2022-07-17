###
 # @Author: kinggreat24
 # @Date: 2022-07-06 02:01:02
 # @LastEditTime: 2022-07-10 20:34:53
 # @LastEditors: kinggreat24
 # @Description: 
 # @FilePath: /ORB_SLAM2/Examples/Monocular-Lidar/run_hybrid_track_kitti.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 

# ./monolidar_kitti ../../Vocabulary/ORBvoc.txt ./KITTI00-02.yaml /media/kinggreat24/Samsung_T5/data/kitti_data_full/odometry/unzip/data/dataset/sequences/$1


# ./monolidar_kitti ../../Vocabulary/ORBvoc.txt ./KITTI03.yaml /media/kinggreat24/Samsung_T5/data/kitti_data_full/odometry/unzip/data/dataset/sequences/$1


./hybrid_tracking_example ../../Vocabulary/ORBvoc.txt ./KITTI04-12.yaml /media/kinggreat24/Samsung_T5/data/kitti_data_full/odometry/unzip/data/dataset/sequences/$1

