import csv
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import transforms3d.euler as t3d
import transforms3d.axangles as t3d_axang
import helper
# import tensorflow as tf
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')




###################### Print Operations #########################
def print_(text="Test", color='w', style='no', bg_color=''):
    color_dict = {'b': 30, 'r': 31, 'g': 32, 'y': 33, 'bl': 34, 'p': 35, 'c': 36, 'w': 37}
    style_dict = {'no': 0, 'bold': 1, 'underline': 2, 'neg1': 3, 'neg2': 5}
    bg_color_dict = {'b': 40, 'r': 41, 'g': 42, 'y': 43, 'bl': 44, 'p': 45, 'c': 46, 'w': 47}
    if bg_color is not '':
        print("\033[" + str(style_dict[style]) + ";" + str(color_dict[color]) + ";" + str(bg_color_dict[bg_color]) + "m" + text + "\033[00m")
    else: print("\033["+ str(style_dict[style]) + ";" + str(color_dict[color]) + "m"+ text + "\033[00m")

###################### Data Downloading Operations #########################

def download_data(file):
    from google_drive_downloader import GoogleDriveDownloader as gdd

    if file=='train_data':
        file_id = '16YU-tdayVNBwM3XlPDgFrrzlPjhQN3PB'
    elif file=='car_data':
        file_id = '1k9W75uhUFTfA_iK7YePGn5t9f4JhtgSe'

    if not os.path.exists(os.path.join(os.getcwd(),'data',file)):
        gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=os.path.join(os.getcwd(),'data',file+'.zip'),
                                        showsize=True,
                                        unzip=True)

        os.remove(os.path.join(os.getcwd(),'data',file+'.zip'))


###################### Data Handling Operations #########################

# Read the templates from a given file.
def read_templates(file_name,templates_dict):
    with open(os.path.join('data',templates_dict,file_name),'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data = []
        for row in csvreader:
            row = [float(i) for i in row]
            data.append(row)
    return data                                         # n2 x 2048 x 3

# Read the file names having templates.
def template_files(templates_dict):
    with open(os.path.join('data',templates_dict,'template_filenames.txt'),'r') as file:
        files = file.readlines()
    files = [x.strip() for x in files]
    print(files)
    return files                                        # 1 x n1

# Read the templates from each file.
def templates_data(templates_dict):
    files = template_files(templates_dict)                          # Read the available file names.
    data = []
    for i in range(len(files)):
        temp = read_templates(files[i],templates_dict)
        for i in temp:
            data.append(i)
    return np.asarray(data)                             # (n1 x n2 x 2048 x 3) & n = n1 x n2

# Preprocess the templates and rearrange them.
def process_templates(templates_dict):
    data = templates_data(templates_dict)                               # Read all the templates.
    print(data.shape[0]/2048)
    templates = []
    for i in range(data.shape[0]/2048):
        start_idx = i*2048
        end_idx = (i+1)*2048
        templates.append(data[start_idx:end_idx,:])
    return np.asarray(templates)                        # Return all the templates (n x 2048 x 3)

# Read poses from given file.
def read_poses(templates_dict, filename):
    # Arguments:
        # filename:         Read data from a given file (string)
    # Output:
        # poses:            Return array of all the poses in the file (n x 6)

    with open(os.path.join('data',templates_dict,filename),'r') as csvfile:
        csvreader = csv.reader(csvfile)
        poses = []
        for row in csvreader:
            row = [float(i) for i in row]
            poses.append(row)
    return np.asarray(poses)

# Read names of files in given data_dictionary.
def read_files(data_dict):
    with open(os.path.join('data',data_dict,'files.txt')) as file:
        files = file.readlines()
        files = [x.split()[0] for x in files]
    return files[0]

# Read data from h5 file and return as templates.
def read_h5(file_name):
    import h5py
    f = h5py.File(file_name, 'r')
    templates = np.array(f.get('templates'))
    f.close()
    return templates

def read_noise_data(data_dict):
    import h5py
    f = h5py.File(os.path.join('data',data_dict,'noise_data.h5'), 'r')
    templates = np.array(f.get('templates'))
    sources = np.array(f.get('sources'))
    f.close()
    return templates, sources

def read_pairs(data_dict, file_name):
    with open(os.path.join('data', data_dict, file_name), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        pairs = []
        for row in csvreader:
            row = [int(x) for x in row]
            pairs.append(row)
    return np.asarray(pairs)

# Main function to load data and return as templates array.
def loadData(data_dict):
    files = read_files(data_dict)   # Read file names.
    print(files)
    templates = read_h5(files)      # Read templates from h5 file using given file_name.
    return templates

def read_modelnet40(data_dict):
    import h5py
    file = h5py.File(os.path.join('data',data_dict,'modelnet40_train.h5'))
    templates = file.get('data')
    return templates

def read_partial_data(data_dict, file):
    import h5py
    f = h5py.File(os.path.join('data',data_dict,file), 'r')
    templates = np.array(f.get('templates'))
    sources = np.array(f.get('sources'))
    poses = np.array(f.get('poses'))
    f.close()
    return templates, sources, poses    

def read_sparse_data(data_dict, file):
    import h5py
    f = h5py.File(os.path.join('data',data_dict,file), 'r')
    templates = np.array(f.get('uniform_data'))
    sources = np.array(f.get('data'))

    for i in range(templates.shape[0]):
        templates[i] = templates[i]/(np.max(templates[i]) - np.min(templates[i]))
        templates[i] = templates[i] - np.mean(templates[i], axis=0, keepdims=True)
    return templates, sources

###################### Transformation Operations #########################

def rotate_point_cloud_by_angle_y(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T       # Pre-Multiplication (changes done)
    return rotated_data

def rotate_point_cloud_by_angle_x(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        shape_pc = batch_data[k, ...]
        # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T       # Pre-Multiplication (changes done)
    return rotated_data

def rotate_point_cloud_by_angle_z(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T       # Pre-Multiplication (changes done)
    return rotated_data

# Translate the data as per given translation vector.
def translate(data,shift):
    # Arguments:
        # data:                 Point Cloud data (1 x num_points x 3)
        # shift:                Translation vector (1 x 3)

    try:
        data = np.asarray(data)
    except:
        pass
    return data+shift

# Apply the given transformation to given point cloud data.
def apply_transformation(datas,poses):          # Transformation function for (2 & 4c, loss 8b)
    # Arguments:
        # datas:            Point Clouds (batch_size x num_points x 3)
        # poses:            translation+euler (Batch_size x 6)
    # Output:
        # transformed_data: Transformed Point Clouds by given poses (batch_size x num_points x 3)
    transformed_data = np.copy(datas)
    for i in range(datas.shape[0]):
        transformed_data[i,:,:] = rotate_point_cloud_by_angle_z(transformed_data[i,:,:],poses[i,5])
        transformed_data[i,:,:] = rotate_point_cloud_by_angle_y(transformed_data[i,:,:],poses[i,4])
        transformed_data[i,:,:] = rotate_point_cloud_by_angle_x(transformed_data[i,:,:],poses[i,3])
        transformed_data[i,:,:] = translate(transformed_data[i,:,:],[poses[i,0],poses[i,1],poses[i,2]])
    return transformed_data

# Convert poses from 6D to 7D.          # For loss function ( 8a )
def poses_euler2quat(poses):
    # Arguments:
        # poses:            6D pose (translation + euler) (batch_size x 6)
    # Output: 
        # new_poses:        7D pose (translation + quaternions) (batch_size x 7)

    new_poses = []                  # Store 7D poses
    for i in range(poses.shape[0]):
        temp = t3d.euler2quat(poses[i,3],poses[i,4],poses[i,5])                     # Convert from euler to quaternion. (1x4)
        temp1 = [poses[i,0],poses[i,1],poses[i,2],temp[0],temp[1],temp[2],temp[3]]      # Add translation & Quaternion (1x7)
        new_poses.append(temp1)                                             
    return np.asarray(new_poses)                                    

# Geenerate random poses equal to batch_size.
def generate_poses(batch_size):
    # Arguments:
        # batch_size:       No of 6D poses required.
    # Output:
        # poses:            Array of poses with translation and rotation (euler angles in radians) (batch_size x 6)

    poses = []                  # List to store the 6D poses.
    for i in range(batch_size):
        # Generate random translations.
        x = np.round(2*np.random.random_sample()-1,2)
        y = np.round(2*np.random.random_sample()-1,2)
        z = np.round(2*np.random.random_sample()-1,2)
        # Generate random rotations.
        x_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
        y_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
        z_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
        poses.append([x,y,z,x_rot,y_rot,z_rot])
    return np.array(poses).reshape((batch_size,6))

# Convert 6D poses to transformation matrix.    # (for 4b)
def transformation(poses):      
    # Arguments:
        # poses:                    6D (x,y,z,euler_x,euler_y,euler_z) (in radians)
    # Output
        # transformation_matrix:    batch_size x 4 x 4

    transformation_matrix = np.zeros((poses.shape[0],4,4))      
    transformation_matrix[:,3,3] = 1
    for i in range(poses.shape[0]):
        rot = t3d.euler2mat(poses[i,5],poses[i,4],poses[i,3],'szyx')    # Calculate rotation matrix using transforms3d
        transformation_matrix[i,0:3,0:3]=rot                            # Store rotation matrix in transformation matrix.
        transformation_matrix[i,0:3,3]=poses[i,0:3]                     # Store translations in transformation matrix.
    return transformation_matrix

# Convert poses (quaternions) to transformation matrix and apply on point cloud.
def transformation_quat2mat(poses,TRANSFORMATIONS,templates_data):      # (for 4b)
    # Arguments:
        # poses:                    7D (x,y,z,quat_q0,quat_q1,quat_q2,quat_q3) (in radians) (batch_size x 7)
        # TRANSFORMATIONS:          Overall tranformation matrix.
        # template_data:            Point Cloud (batch_size x num_points x 3)
    # Output
        # TRANSFORMATIONS:          Batch_size x 4 x 4
        # templates_data:           Transformed template data (batch_size x num_points x 3)
    
    poses = np.array(poses)                                             # Convert poses to array.
    poses = poses.reshape(poses.shape[-2],poses.shape[-1])
    for i in range(poses.shape[0]):
        transformation_matrix = np.zeros((4,4))
        transformation_matrix[3,3] = 1
        rot = t3d.quat2mat([poses[i,3],poses[i,4],poses[i,5],poses[i,6]])   # Calculate rotation matrix using transforms3d
        transformation_matrix[0:3,0:3]=rot                                  # Store rotation matrix in transformation matrix.
        transformation_matrix[0:3,3]=poses[i,0:3]                           # Store translations in transformation matrix.
        TRANSFORMATIONS[i,:,:] = np.dot(transformation_matrix,TRANSFORMATIONS[i,:,:])       # 4b (Multiply tranfromation matrix to Initial Transfromation Matrix)
        templates_data[i,:,:]=np.dot(rot,templates_data[i,:,:].T).T         # Apply Rotation to Template Data
        templates_data[i,:,:]=templates_data[i,:,:]+poses[i,0:3]            # Apply translation to template data
    return TRANSFORMATIONS,templates_data

# Convert the Final Transformation Matrix to Translation + Orientation (Euler Angles in Degrees)
def find_final_pose(TRANSFORMATIONS):
    # Arguments:
        # TRANSFORMATIONS:          transformation matrix (batch_size x 4 x 4)
    # Output:
        # final_pose:               final pose predicted by network (batch_size x 6)

    final_pose = np.zeros((TRANSFORMATIONS.shape[0],6))     # Array to store the poses.
    for i in range(TRANSFORMATIONS.shape[0]):               
        rot = TRANSFORMATIONS[i,0:3,0:3]                    # Extract rotation matrix.
        euler = t3d.mat2euler(rot,'szyx')                   # Convert rotation matrix to euler angles. (Pre-multiplication)
        final_pose[i,3:6]=[euler[2],euler[1],euler[0]]      # Store the translation
        final_pose[i,0:3]=TRANSFORMATIONS[i,0:3,3].T        # Store the euler angles.
    return final_pose

# Convert the Final Transformation Matrix to Translation + Orientation (Euler Angles in Degrees)
def find_final_pose_inv(TRANSFORMATIONS_ip):
    # Arguments:
        # TRANSFORMATIONS:          transformation matrix (batch_size x 4 x 4)
    # Output:
        # final_pose:               final pose predicted by network (batch_size x 6)

    TRANSFORMATIONS = np.copy(TRANSFORMATIONS_ip)
    final_pose = np.zeros((TRANSFORMATIONS.shape[0],6))     # Array to store the poses.
    for i in range(TRANSFORMATIONS.shape[0]):
        TRANSFORMATIONS[i] = np.linalg.inv(TRANSFORMATIONS[i])              
        rot = TRANSFORMATIONS[i,0:3,0:3]                    # Extract rotation matrix.
        euler = t3d.mat2euler(rot,'szyx')                   # Convert rotation matrix to euler angles. (Pre-multiplication)
        final_pose[i,3:6]=[euler[2],euler[1],euler[0]]      # Store the translation
        final_pose[i,0:3]=TRANSFORMATIONS[i,0:3,3].T        # Store the euler angles.
    return final_pose

# Subtract the centroids from source and template (Like ICP) and then find the pose.
def centroid_subtraction(source_data, template_data):
    # Arguments:
        # source_data:          Source Point Clouds (batch_size x num_points x 3)
        # template_data:        Template Point Clouds (batch_size x num_points x 3)
    # Output:
        # source_data:                  Centroid subtracted from source point cloud (batch_size x num_points x 3)
        # template_data:                Centroid subtracted from template point cloud (batch_size x num_points x 3)
        # centroid_translation_pose:    Apply this pose after final iteration. (batch_size x 7)

    centroid_translation_pose = np.zeros((source_data.shape[0],7))
    for i in range(source_data.shape[0]):
        source_centroid = np.mean(source_data[i],axis=0)
        template_centroid = np.mean(template_data[i],axis=0)
        source_data[i] = source_data[i] - source_centroid
        template_data[i] = template_data[i] - template_centroid
        centroid_translation = source_centroid - template_centroid
        centroid_translation_pose[i] = np.array([centroid_translation[0],centroid_translation[1],centroid_translation[2],1,0,0,0])
    return source_data, template_data, centroid_translation_pose

def inverse_pose(pose):
    transformation_pose = np.zeros((4,4))
    transformation_pose[3,3]=1
    transformation_pose[0:3,0:3] = t3d.euler2mat(pose[5], pose[4], pose[3], 'szyx')
    transformation_pose[0,3] = pose[0]
    transformation_pose[1,3] = pose[1]
    transformation_pose[2,3] = pose[2]
    transformation_pose = np.linalg.inv(transformation_pose)
    pose_inv = np.zeros((1,6))[0]
    pose_inv[0] = transformation_pose[0,3]
    pose_inv[1] = transformation_pose[1,3]
    pose_inv[2] = transformation_pose[2,3]
    orient_inv = t3d.mat2euler(transformation_pose[0:3,0:3], 'szyx')
    pose_inv[3] = orient_inv[2]
    pose_inv[4] = orient_inv[1]
    pose_inv[5] = orient_inv[0]
    return pose_inv

def pose2mat(poses):
    mat = np.zeros([poses.shape[0],4,4])
    for idx, pose in enumerate(poses):
        mat[idx,0:3,0:3] = t3d.euler2mat(pose[5],pose[4],pose[3],'szyx')
        mat[idx,0:3,3] = pose[0:3]
    return mat


###################### Shuffling Operations #########################

# Randomly shuffle given array of poses for training procedure.
def shuffle_templates(templates):
    # Arguments:
        # templates:            Input array of templates to get randomly shuffled (batch_size x num_points x 3)
    # Output:
        # shuffled_templates:   Randomly ordered poses (batch_size x num_points x 3)

    shuffled_templates = np.zeros(templates.shape)                      # Array to store shuffled templates.
    templates_idxs = np.arange(0,templates.shape[0])
    np.random.shuffle(templates_idxs)                                   # Randomly shuffle template indices.
    for i in range(templates.shape[0]):
        shuffled_templates[i,:,:]=templates[templates_idxs[i],:,:]      # Rearrange them as per shuffled indices.
    return shuffled_templates

# Randomly shuffle given array of poses for training procedure.
def shuffle_poses(poses):
    # Arguments:
        # poses:            Input array of poses to get randomly shuffled (batch_size x n)
    # Output:
        # shuffled_poses:   Randomly ordered poses (batch_size x n)

    shuffled_poses = np.zeros(poses.shape)              # Array to store shuffled poses.
    poses_idxs = np.arange(0,poses.shape[0])            
    np.random.shuffle(poses_idxs)                       # Shuffle the indexes of poses.
    for i in range(poses.shape[0]):
        shuffled_poses[i,:]=poses[poses_idxs[i],:]      # Rearrange them as per shuffled indexes.
    return shuffled_poses

# Generate random transformation/pose for data augmentation.
def random_trans():
    # Output:
        # 6D pose with first 3 translation values and last 3 euler angles in radian about x,y,z-axes. (1x6)

    # Generate random translations.
    x_trans, y_trans, z_trans = 0.4*np.random.uniform()-0.2, 0.4*np.random.uniform()-0.2, 0.4*np.random.uniform()-0.2   
    # Generate random rotation angles.
    x_rot, y_rot, z_rot = (np.pi/9)*np.random.uniform()-(np.pi/18), (np.pi/9)*np.random.uniform()-(np.pi/18), (np.pi/9)*np.random.uniform()-(np.pi/18)
    return [x_trans,y_trans,z_trans,x_rot,y_rot,z_rot]

# Generate random poses for each batch to train the network.
def generate_random_poses(batch_size):
    # Arguments:
        # Batch_size:       No of poses in the output
    # Output:
        # poses:            Randomly generated poses (batch_size x 6)

    poses = []
    for i in range(batch_size):
        x_trans, y_trans, z_trans = 2*np.random.uniform()-1, 2*np.random.uniform()-1, 2*np.random.uniform()-1                                       # Generate random translation
        x_rot, y_rot, z_rot = (np.pi)*np.random.uniform()-(np.pi/2), (np.pi)*np.random.uniform()-(np.pi/2), (np.pi)*np.random.uniform()-(np.pi/2)   # Generate random orientation
        poses.append([np.round(x_trans,4), np.round(y_trans,4), np.round(z_trans,4), np.round(x_rot,4), np.round(y_rot,4), np.round(z_rot,4)])      # round upto 4 decimal digits
    return np.array(poses)

def select_random_points(source_data, num_point):
    random_source_data = np.copy(source_data)
    idx = np.arange(random_source_data.shape[1])            # Find indexes of source data.
    np.random.shuffle(idx)                                  # Shuffle indexes.
    random_source_data = random_source_data[:,idx,:]        # Shuffle data as per shuffled indexes.
    return random_source_data[:,0:num_point,:]

def add_noise(source_data):
    for i in range(source_data.shape[0]):
        mean = 0
        for j in range(source_data.shape[1]):
            sigma = 0.04*np.random.random_sample()              # Generate random variance value b/w 0 to 0.1
            source_data[i,j,:] = source_data[i,j,:] + np.random.normal(mean, sigma, source_data[i,j].shape) # Add gaussian noise.
    return source_data

def fit_in_m1_to_1(points):
    '''
    Input: Nx3 
    Output: Nx3 
    fits the point cloud in [(-1,-1,-1) to (1,1,1)]
    '''
    points = points - np.mean(points,axis=0)
    dist_from_orig = np.linalg.norm(points,axis=1)
    points = points/np.max(dist_from_orig)
    return points

###################### Tensor Operations #########################

def rotate_point_cloud_by_angle_y_tensor(data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    cosval = tf.cos(rotation_angle)
    sinval = tf.sin(rotation_angle)
    rotation_matrix = tf.reshape([[cosval, 0, sinval],[0, 1, 0],[-sinval, 0, cosval]], [3,3])
    data = tf.reshape(data, [-1, 3])
    rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
    return rotated_data

def rotate_point_cloud_by_angle_x_tensor(data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    cosval = tf.cos(rotation_angle)
    sinval = tf.sin(rotation_angle)
    rotation_matrix = tf.reshape([[1, 0, 0],[0, cosval, -sinval],[0, sinval, cosval]], [3,3])
    data = tf.reshape(data, [-1, 3])
    rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
    return rotated_data

def rotate_point_cloud_by_angle_z_tensor(data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    cosval = tf.cos(rotation_angle)
    sinval = tf.sin(rotation_angle)
    rotation_matrix = tf.reshape([[cosval, -sinval, 0],[sinval, cosval, 0],[0, 0, 1]], [3,3])
    data = tf.reshape(data, [-1, 3])
    rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
    return rotated_data

def translate_tensor(data,shift):
    # Add the translation vector to given tensor. (num_point x 3)
    return tf.add(data,shift)   

# Tranform the data as per given poses with orientation as euler in degrees.
def transformation_tensor(datas,poses):
    # Arguments:
        # datas:                Tensor of Point Cloud (batch_size x num_points x 3)
        # poses:                Tensor of Poses (translation + euler angles in degrees) (batch_size x num_points x 3)
    # Ouput:
        # transformed_data:     Tensor of transformed point cloud (batch_size x num_points x 3)

    transformed_data = tf.zeros([datas.shape[1], datas.shape[2]])       # Tensor to store the transformed point clouds as tensor.
    for i in range(datas.shape[0]):
        transformed_data_t = rotate_point_cloud_by_angle_x_tensor(datas[i,...],poses[i,3])              # Rotate about x-axis
        transformed_data_t = rotate_point_cloud_by_angle_y_tensor(transformed_data_t,poses[i,4])        # Rotate about y-axis
        transformed_data_t = rotate_point_cloud_by_angle_z_tensor(transformed_data_t,poses[i,5])        # Rotate about z-axis
        transformed_data_t = translate_tensor(transformed_data_t,[poses[i,0],poses[i,1],poses[i,2]])    # Translate by given vector.
        transformed_data = tf.concat([transformed_data, transformed_data_t], 0)                         # Append the transformed tensor point cloud.
    transformed_data = tf.reshape(transformed_data, [-1, datas.shape[1], datas.shape[2]])[1:]           # Reshape the transformed tensor and remove first one. (batch_size x num_point x 3)
    return transformed_data

# Tranform the data as per given poses with orientation as quaternion.
def transformation_quat_tensor(data,quat,translation):
    # Arguments:
        # data:                 Tensor of Point Cloud. (batch_size x num_point x 3)
        # quat:                 Quaternion tensor to generate rotation matrix.  (batch_size x 4)
        # translation:          Translation tensor to translate the point cloud. (batch_size x 3)
    # Outputs:
        # transformed_data:     Tensor of Rotated and Translated Point Cloud Data. (batch_size x num_points x 3)

    transformed_data = tf.zeros([data.shape[1],3])      # Tensor to store transformed data.
    for i in range(quat.shape[0]):
        # Seperate each quaternion value.
        q0 = tf.slice(quat,[i,0],[1,1])
        q1 = tf.slice(quat,[i,1],[1,1])
        q2 = tf.slice(quat,[i,2],[1,1])
        q3 = tf.slice(quat,[i,3],[1,1])
        # Convert quaternion to rotation matrix.
        # Ref:  http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
              # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
        R = [[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
             [2*(q1*q2+q0*q3), q0*q0+q2*q2-q1*q1-q3*q3, 2*(q2*q3-q0*q1)],
             [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0+q3*q3-q1*q1-q2*q2]]

        R = tf.reshape(R,[3,3])             # Convert R into a single tensor of shape 3x3.
        # tf.tensordot: Arg: tensor1, tensor2, axes
        # axes defined for tensor1 & tensor2 should be of same size.
        # axis 1 of R is of size 3 and axis 0 of data (3xnum_points) is of size 3.

        temp_rotated_data = tf.transpose(tf.tensordot(R, tf.transpose(data[i,...]), [1,0]))     # Rotate the data. (num_points x 3)
        temp_rotated_data = tf.add(temp_rotated_data,translation[i,...])                        # Add the translation (num_points x 3)
        transformed_data = tf.concat([transformed_data, temp_rotated_data],0)                   # Append data (batch_size x num_points x 3)
    transformed_data = tf.reshape(transformed_data, [-1,data.shape[1],3])[1:]                   # Reshape data and remove first point cloud. (batch_size x num_point x 3)
    return transformed_data


###################### Display Operations #########################

# Display data inside ModelNet files.
def display_clouds(filename,model_no):
    # Arguments:
        # filename:         Name of file to read the data from. (string)
        # model_no:         Number to choose the model inside that file. (int)

    data = []
    # Read the entire data from that file.
    with open(os.path.join('data','templates',filename),'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row = [float(x) for x in row]
            data.append(row)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    data = np.asarray(data)

    start_idx = model_no*2048
    end_idx = (model_no+1)*2048
    data = data[start_idx:end_idx,:]        # Choose specific data related to the given model number.

    X,Y,Z = [],[],[]
    for row in data:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    ax.scatter(X,Y,Z)
    plt.show()

# Display given Point Cloud Data in blue color (default).
def display_clouds_data(data):
    # Arguments:
        # data:         array of point clouds (num_points x 3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data = data.tolist()
    except:
        pass
    X,Y,Z = [],[],[]
    for row in data:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    ax.scatter(X,Y,Z)
    ax.set_aspect('equal')
    plt.show()

 

def display_two_clouds(data1,data2,title="title",disp_corr=False):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
         

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[0,1,0,0.5])
    
    if disp_corr:
        for i in range(len(data1)):
            ax.plot3D([data1[i][0],data2[i][0] ], \
                  [data1[i][1],data2[i][1] ], \
                  [data1[i][2],data2[i][2] ], \
                   c=[0,0,1,0.1] )
 

    # Add details to Plot.
    plt.legend((l1,l2),('training data','Source Data' ),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    ax.set_aspect('equal')
    
    plt.show()

def display_two_clouds_corr_mat(data1,data2,corr_mat,text=["title","pointcloud1","pointcloud2"],disp_time=3):
    '''
    Arguments:
        data1         source (num_points x 3) (Red)
        data2         target (num_points x 3) (Green)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    data3 = (data2.T.dot(corr_mat)).T 
    # print (data3,"data3")
 
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=np.array([1,0,0,1]))
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=np.array([0,1,0,1]))

    # X,Y,Z = [],[],[]
    # for row in data3:
    #   X.append(row[0])
    #   Y.append(row[1])
    #   Z.append(row[2])
    # l3 = ax.scatter(X,Y,Z,c=np.array([1,1,0,1]))
 
 
    for i in range(len(data1)):
        if np.random.rand() > 0.6:
            ax.plot3D([data1[i][0],data3[i][0]] , \
                      [data1[i][1],data3[i][1]] , \
                      [data1[i][2],data3[i][2]] , \
                       '--',c=np.array([0,0,1,1])) 
            # ax.plot3D([data1[i][0],data2[corr_idx_gt[i]][0] ], \
            #         [data1[i][1],data2[corr_idx_gt[i]][1] ], \
            #         [data1[i][2],data2[corr_idx_gt[i]][2] ], \
            #          c=np.array([0,0,0,0.5]))
 
    # Add details to Plot.
    plt.legend((l1,l2),(text[1],text[2]),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=5)
    ax.set_ylabel('Y-axis',fontsize=5)
    ax.set_zlabel('Z-axis',fontsize=5)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(text[0],fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    # ax.set_aspect('equal')
    
    plt.show(block=False)
    try:
        plt.pause(disp_time)
        plt.close()
    except:
        pass


def display_two_clouds_correspondence(data1,data2,corr_idx,corr_idx_gt,title="title",disp_time=3):
    '''
    Issue with this function:
    shuffles source data. 
    We generally want to shuffle target data and transform source data.
    '''
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
         

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=np.array([1,0,0,1]))
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=np.array([0,1,0,1]))

    # X,Y,Z = [],[],[]
    # for row in data3:
    #   X.append(row[0])
    #   Y.append(row[1])
    #   Z.append(row[2])
    # l3 = ax.scatter(X,Y,Z,c=np.array([1,1,0,1]))
 
 
    for i in range(len(corr_idx)):
        ax.plot3D([data1[i][0],data2[corr_idx[i]][0] ], \
                  [data1[i][1],data2[corr_idx[i]][1] ], \
                  [data1[i][2],data2[corr_idx[i]][2] ], \
                   '--',c=np.array([0,0,1,1]) )
        # ax.plot3D([data1[i][0],data2[corr_idx_gt[i]][0] ], \
        #         [data1[i][1],data2[corr_idx_gt[i]][1] ], \
        #         [data1[i][2],data2[corr_idx_gt[i]][2] ], \
        #          c=np.array([0,0,0,0.5]))
 
    # Add details to Plot.
    plt.legend((l1,l2),('transformed source','source'),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    ax.set_aspect('equal')
    
    plt.show(block=False)
    try:
        plt.pause(disp_time)
        plt.close()
    except:
        pass     
# Display given template, source and predicted point cloud data.
def display_three_clouds(data1,data2,data3,title,legend_list):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
        # data3         Predicted Data (num_points x 3) (Blue)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[0,0,1,1],s=10)
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[1,0,0,1],s=10)
    # Add Predicted Data in Plot
    X,Y,Z = [],[],[]
    for row in data3:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l3 = ax.scatter(X,Y,Z,c=[0,1,0,0.5],s=30)

    # Add details to Plot.
    plt.legend((l1,l2,l3),(legend_list[0],legend_list[1],legend_list[2]),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    # ax.set_aspect('equal')

    plt.show()


def display_n_clouds(data_list,
                     legend_list,
                     color_list=['y','r','g','k','b','m','c'],
                     title="n_clouds"):
    """
    max 7 colors
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(legend_list)):
        data = data_list[i]
        ax.scatter(data[:,0],data[:,1],data[:,2], 
                   c=color_list[i],
                   s = 30-6*i,
                   alpha = 0.3+i/7.,
                   label=legend_list[i]) 
        
    
    plt.title(title,fontdict={'fontsize':25})
    ax.legend(fontsize=20,loc=1)
    plt.show(block=False)   




# Display template, source, predicted point cloud data with results after each iteration.
def display_itr_clouds(data1,data2,data3,ITR,title):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
        # data3         Predicted Data (num_points x 3) (Blue)
        # ITR           Point Clouds obtained after each iteration (iterations x batch_size x num of points x 3) (Yellow)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    print(ITR.shape)        # Display Number of Point Clouds in ITR.
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[0,1,0,1])
    # Add Predicted Data in Plot
    X,Y,Z = [],[],[]
    for row in data3:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l3 = ax.scatter(X,Y,Z,c=[0,0,1,1])
    # Add point clouds after each iteration in Plot.
    for itr_data in ITR:
        X,Y,Z = [],[],[]
        for row in itr_data[0]:
            X.append(row[0])
            Y.append(row[1])
            Z.append(row[2])
        ax.scatter(X,Y,Z,c=[1,1,0,0.5])

    # Add details to Plot.
    plt.legend((l1,l2,l3),('Template Data','Source Data','Predicted Data'),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    plt.show()

def display_rotation_vects_SO3(rot_vects):
    # print (rot_vects)
    # print (rot_vects.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.multiply(rot_vects[:,0],rot_vects[:,3])
    ys = np.multiply(rot_vects[:,1],rot_vects[:,3])
    zs = np.multiply(rot_vects[:,2],rot_vects[:,3])

    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def rotate_point_cloud_by_axis_angle(batch_data,axis,rotation_angle):
    '''
    input : BxNx3 , 1x3 vect,angle
    output: BxNx3
    '''
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi 
        rotation_matrix = t3d_axang.axangle2mat(axis, rotation_angle)

        shape_pc = batch_data[k, ...]
        # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T       # Pre-Multiplication (changes done)
    return rotated_data


# Log test results to a folder
def invert_axang(axis,rotation_angle):
    rot_m = t3d_axang.axangle2mat(axis, rotation_angle)
    axis_inv,angle_inv = t3d_axang.mat2axangle(rot_m.T)
    return axis_inv,angle_inv
    
def combine_ax_angs(rot_vect_seq):
    R = np.eye(3)
    for i in range(len(rot_vect_seq)):
        R = R.dot(t3d_axang.axangle2mat(rot_vect_seq[i,0:3], rot_vect_seq[i,3]))
        # print R
    comb_axis_inv,comb_angle_inv = t3d_axang.mat2axangle(R)
    return  comb_axis_inv,comb_angle_inv


## SO3 related helper functions
def distance_between_ax_angs(v1,v2):
    """
    v1 = [ax_x,ax_y,ax_z,ang]
    v2 = [ax_x,ax_y,ax_z,ang]
    """
 
    R1 = t3d_axang.axangle2mat(v1[0:3],v1[3],is_normalized=True)
    R2 = t3d_axang.axangle2mat(v2[0:3],v2[3],is_normalized=True)

    ax,ang = t3d_axang.mat2axangle(R1.dot(R2.T))
    return abs(ang)

def find_nn_SO3 (rot_vects, query_vect):
    n = len(rot_vects)
    closest_sample_id_SO3 = 0
    min_ang = np.pi
    for i in range(n):
        cur_dist = distance_between_ax_angs(rot_vects[i,:],query_vect)
        if  cur_dist < min_ang:
            min_ang = cur_dist
            closest_sample_id_SO3 = i

    return  closest_sample_id_SO3

def display_SO3(rot_vects,label = 'training set SO3 points'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = np.multiply(rot_vects[:,0],rot_vects[:,3])
    ys = np.multiply(rot_vects[:,1],rot_vects[:,3])
    zs = np.multiply(rot_vects[:,2],rot_vects[:,3])

    ax.scatter(xs, ys, zs,s = 5,label=label)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend(fontsize=20,loc=1)
    plt.show(block=False)


def sample_from_SO3(dtheta=np.pi/6,dr=0.2,dphi=np.pi/6):
    '''
    output: rot_vects = nx4 matrix of rotation vectors 
            rot_vects[i,0:3] = normalized axis 
            rot_vects[i,3] = angle of rotation 
    '''

    theta = np.arange(0,np.pi ,dtheta)
    r = np.arange(dr,np.pi+dr,dr)**3/((np.pi)**2)
    phi = np.arange(0,2*np.pi,dphi)
        
    n_theta = len(theta)
    n_r = len(r)
    n_phi = len(phi)

    # print theta,"theta"
    # print phi,"phi"
    # print r,"r"
 
    rot_vects = np.zeros(((n_theta*n_r*n_phi) , 4)) 
 
    theta_rep = np.hstack((theta.reshape(n_theta,1),)*n_r*n_phi )
    theta_rep = theta_rep.reshape(n_phi*n_r*n_theta,1)
    
    r_rep = np.hstack((r.reshape(n_r,1),)*n_phi)
    r_rep = np.vstack((r_rep,)*n_theta)
    r_rep = r_rep.reshape(n_phi*n_r*n_theta,1)

    phi_rep = np.hstack((phi.T.reshape(n_phi,1),)*n_r*n_theta )
    phi_rep = phi_rep.T.reshape(n_phi*n_r*n_theta,1)
    rot_params = np.hstack((theta_rep,r_rep,phi_rep))
 
    rot_vects[:,0] = np.multiply(np.sin(rot_params[:,0]),np.cos(rot_params[:,2]))
    rot_vects[:,1] = np.multiply(np.sin(rot_params[:,0]),np.sin(rot_params[:,2]))
    rot_vects[:,2] = np.cos(rot_params[:,0])
    rot_vects[:,3] = rot_params[:,1]
    rot_vects=np.unique(rot_vects,axis=0)
    
    norm_vect = np.linalg.norm(rot_vects[:,0:3],axis=1).reshape(len(rot_vects),1)
    rot_vects[:,0:3] = np.divide(rot_vects[:,0:3],norm_vect)
    rot_vects=np.unique(rot_vects,axis=0)
     
 

### debugging helper 
    # print (rot_vects)
    # print (rot_vects.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # xs = np.multiply(rot_vects[:,0],rot_vects[:,3])
    # ys = np.multiply(rot_vects[:,1],rot_vects[:,3])
    # zs = np.multiply(rot_vects[:,2],rot_vects[:,3])

 #  ax.scatter(xs, ys, zs)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    # helper.display_rotation_vects_SO3(rot_vects)
    return rot_vects




def log_test_results(LOG_DIR, filename, log):

    # It will log the data in following sequence in csv format:
    # Sr. No., time taken, number of iterations, translation error, rotation error.

    # If log dir doesn't exists create one.
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

    # Find params from the dictionary.
    ITR, TIME = log['ITR'], log['TIME']
    Trans_Err, Rot_Err = log['Trans_Err'], log['Rot_Err']
    idxs_5_5, idxs_10_1, idxs_20_2 = log['idxs_5_5'], log['idxs_10_1'], log['idxs_20_2']
    num_batches = log['num_batches']

    # Find mean and variance.
    TIME_mean = sum(TIME)/len(TIME)

    # Log all the data in a csv file.
    import csv
    with open(os.path.join(LOG_DIR, filename+'.csv'),'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(len(TIME)):
            csvwriter.writerow([i, TIME[i], ITR[i], Trans_Err[i], Rot_Err[i]])

    if len(idxs_5_5) != 0:
        accuray_5_5 = len(idxs_5_5)/(num_batches*1.0)
        mean_5_5_rot_err = np.sum(np.array(Rot_Err)[idxs_5_5])/len(idxs_5_5)
        var_5_5_rot_err = np.var(np.array(Rot_Err)[idxs_5_5])
        mean_5_5_trans_err = np.sum(np.array(Trans_Err)[idxs_5_5])/len(idxs_5_5)
        var_5_5_trans_err = np.var(np.array(Trans_Err)[idxs_5_5])
        mean_5_5_itr = np.sum(np.array(ITR)[idxs_5_5])/len(idxs_5_5)
        var_5_5_itr = np.var(np.array(ITR)[idxs_5_5])
        mean_5_5_time = np.sum(np.array(TIME)[idxs_5_5])/len(idxs_5_5)
        var_5_5_time = np.var(np.array(TIME)[idxs_5_5])
    else:
        accuray_5_5, mean_5_5_rot_err, var_5_5_rot_err, mean_5_5_trans_err, var_5_5_trans_err, mean_5_5_itr, var_5_5_itr, mean_5_5_time, var_5_5_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

    if len(idxs_10_1) != 0:
        accuray_10_1 = len(idxs_10_1)/(num_batches*1.0)
        mean_10_1_rot_err = np.sum(np.array(Rot_Err)[idxs_10_1])/len(idxs_10_1)
        var_10_1_rot_err = np.var(np.array(Rot_Err)[idxs_10_1])
        mean_10_1_trans_err = np.sum(np.array(Trans_Err)[idxs_10_1])/len(idxs_10_1)
        var_10_1_trans_err = np.var(np.array(Trans_Err)[idxs_10_1])
        mean_10_1_itr = np.sum(np.array(ITR)[idxs_10_1])/len(idxs_10_1)
        var_10_1_itr = np.var(np.array(ITR)[idxs_10_1])
        mean_10_1_time = np.sum(np.array(TIME)[idxs_10_1])/len(idxs_10_1)
        var_10_1_time = np.var(np.array(TIME)[idxs_10_1])
    else:
        accuray_10_1, mean_10_1_rot_err, var_10_1_rot_err, mean_10_1_trans_err, var_10_1_trans_err, mean_10_1_itr, var_10_1_itr, mean_10_1_time, var_10_1_time = 0, 0, 0, 0, 0, 0, 0, 0, 0      
        
    if len(idxs_20_2) != 0:
        # Find accuracies:
        accuray_20_2 = len(idxs_20_2)/(num_batches*1.0)
        # Find mean rotation error.
        mean_20_2_rot_err = np.sum(np.array(Rot_Err)[idxs_20_2])/len(idxs_20_2)
        # Find variance of rotation error.
        var_20_2_rot_err = np.var(np.array(Rot_Err)[idxs_20_2])
        # Find mean translation error.
        mean_20_2_trans_err = np.sum(np.array(Trans_Err)[idxs_20_2])/len(idxs_20_2)
        # Find variance of translation error.
        var_20_2_trans_err = np.var(np.array(Trans_Err)[idxs_20_2])
        # Find mean iterations.
        mean_20_2_itr = np.sum(np.array(ITR)[idxs_20_2])/len(idxs_20_2)
        # Find variance of iterations.
        var_20_2_itr = np.var(np.array(ITR)[idxs_20_2])
        # Find mean time required.
        mean_20_2_time = np.sum(np.array(TIME)[idxs_20_2])/len(idxs_20_2)
        # Find variance of time.
        var_20_2_time = np.var(np.array(TIME)[idxs_20_2])
    else:
        accuray_20_2, mean_20_2_rot_err, var_20_2_rot_err, mean_20_2_trans_err, var_20_2_trans_err, mean_20_2_itr, var_20_2_itr, mean_20_2_time, var_20_2_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

    with open(os.path.join(LOG_DIR, filename+'.txt'),'w') as file:
        file.write("Mean of Time: {}\n".format(TIME_mean))

        file.write("\n")

        file.write("###### 5 Degree & 0.05 Units ######\n")
        file.write("Accuray: {}%\n".format(accuray_5_5*100))
        file.write("Mean rotational error: {}\n".format(mean_5_5_rot_err))
        file.write("Mean translation error: {}\n".format(mean_5_5_trans_err))
        file.write("Mean time: {}\n".format(mean_5_5_time))
        file.write("Var time: {}\n".format(var_5_5_time))
        file.write("Var translation error: {}\n".format(var_5_5_trans_err))
        file.write("Var rotational error: {}\n".format(var_5_5_rot_err))
        file.write("Mean Iterations: {}\n".format(mean_5_5_itr))
        file.write("Var Iterations: {}\n".format(var_5_5_itr))

        file.write("\n")

        file.write("###### 10 Degree & 0.1 Units ######\n")
        file.write("Accuray: {}%\n".format(accuray_10_1*100))
        file.write("Mean rotational error: {}\n".format(mean_10_1_rot_err))
        file.write("Mean translation error: {}\n".format(mean_10_1_trans_err))
        file.write("Mean time: {}\n".format(mean_10_1_time))
        file.write("Var time: {}\n".format(var_10_1_time))
        file.write("Var translation error: {}\n".format(var_10_1_trans_err))
        file.write("Var rotational error: {}\n".format(var_10_1_rot_err))
        file.write("Mean Iterations: {}\n".format(mean_10_1_itr))
        file.write("Var Iterations: {}\n".format(var_10_1_itr))

        file.write("\n")

        file.write("###### 20 Degree & 0.2 Units ######\n")
        file.write("Accuray: {}%\n".format(accuray_20_2*100))
        file.write("Mean rotational error: {}\n".format(mean_20_2_rot_err))
        file.write("Mean translation error: {}\n".format(mean_20_2_trans_err))
        file.write("Mean time: {}\n".format(mean_20_2_time))
        file.write("Var time: {}\n".format(var_20_2_time))
        file.write("Var translation error: {}\n".format(var_20_2_trans_err))
        file.write("Var rotational error: {}\n".format(var_20_2_rot_err))
        file.write("Mean Iterations: {}\n".format(mean_20_2_itr))
        file.write("Var Iterations: {}\n".format(var_20_2_itr))

    plt.hist(Rot_Err,np.arange(0,185,5))
    plt.xlim(0,180)
    plt.savefig(os.path.join(LOG_DIR,'rot_err_hist.jpeg'),dpi=500,quality=100)
    plt.figure()
    plt.hist(Trans_Err,np.arange(0,1.01,0.01))
    plt.xlim(0,1)
    plt.savefig(os.path.join(LOG_DIR,'trans_err_hist.jpeg'),dpi=500,quality=100)



def points_above_plane(points,plane_coeff):
    '''
        points: nx3
        plane_coeff: [a,b,c,d] of ax+by+cz+d=0
    '''
    if len(plane_coeff) ==3:
        a=plane_coeff[0]
        b=plane_coeff[1]
        c=plane_coeff[2]
        d=0
    else:
        a=plane_coeff[0]
        b=plane_coeff[1]
        c=plane_coeff[2]
        d=plane_coeff[3]


    distances = a*points[:,0] + b*points[:,1] + c*points[:,2] +d
    above_idx = np.flip(np.argsort(distances ))
    # print  (above_idx,"above_idx")
    above_points = points[above_idx,:]

    return above_points,above_idx

def display_partial_with_plane(points,above_points,plane_coeff):

    if len(plane_coeff) ==3:
        a=plane_coeff[0]
        b=plane_coeff[1]
        c=plane_coeff[2]
        d=0
    else:
        a=plane_coeff[0]
        b=plane_coeff[1]
        c=plane_coeff[2]
        d=plane_coeff[3]

    p1 = [1,1,-(a+b+d)/c] 
    p2 = [1,-1,-(a-b+d)/c] 
    p3 = [-1,-1,-(-a-b+d)/c] 
    p4 = [-1,1,-(-a+b+d)/c]
    corners = np.array([p1,p2,p3,p4,p1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
     
    ax.scatter(points[:,0], points[:,1], points[:,2],c='r',s = 20 )
    ax.scatter(above_points[:,0], above_points[:,1], above_points[:,2],c='b',s=10 )
    ax.plot(corners[:,0],corners[:,1],corners[:,2])
    ax.plot([0,a*2],[0,b*2],[0,c*2],c='g',linewidth=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.axis('equal')
    ax.set_aspect('equal')
    plt.show()

def print_rounded(a,name='name',decimals=2):
    print(np.around(a,decimals=decimals), name)


if __name__=='__main__':
    # a = np.array([[0,0,0,0,0,0],[0,0,0,90,0,0]])
    # print a.shape
    # a = poses_euler2quat(a)
    # print(a[1,3]*a[1,3]+a[1,4]*a[1,4]+a[1,5]*a[1,5]+a[1,6]*a[1,6])
    # print(a[0,3]*a[0,3]+a[0,4]*a[0,4]+a[0,5]*a[0,5]+a[0,6]*a[0,6])
    # print a.shape
    # display_clouds('airplane_templates.csv',0)

    templates = helper.process_templates('multi_model_templates')
    # templates = helper.process_templates('templates')
    # airplane = templates[0,:,:]
    idx = 199
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # start = idx*2048
    # end = (idx+1)*2048
    ax.scatter(templates[idx,:,0],templates[idx,:,1],templates[idx,:,2])
    plt.show()
    print(templates.shape)

