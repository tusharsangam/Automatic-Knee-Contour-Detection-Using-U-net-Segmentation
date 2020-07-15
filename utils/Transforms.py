import numpy as np
#import SimpleITK as sitk
import copy
import random
import math
import torch
import cv2
from skimage import img_as_float

def trunc_images(img, landmarks, frame_size=800):
    max_x, min_x = landmarks[:,0].max(), landmarks[:,0].min()
    max_y, min_y = landmarks[:,1].max(), landmarks[:,1].min()
    dist_x = max_x - min_x
    dist_y = max_y - min_y
    w, h = img.shape[::-1]
    if dist_x <= frame_size:
        margin_x = int((frame_size - dist_x)/2)
    else :
        margin_x = 0
    
    start_x = int(min_x - margin_x)
    
    if int(max_x + margin_x + 1) <= w:
        end_x = int(max_x + margin_x + 1)
    else:
        end_x = w
    
    if dist_y <= frame_size:
        margin_y = int((frame_size - dist_y)/2)
    else:
        margin_y = 0
    
    if min_y - margin_y >= 0:
        start_y = int(min_y - margin_y)
    else:
        start_y = 0
    
    end_y = int(max_y + margin_y + 1)
    landmarks[:,0] -= start_x
    landmarks[:,1] -= start_y
    return img[start_y:end_y, start_x:end_x], landmarks

def find_quadratic_subpixel_maximum_in_image(image):
    max_index = np.argmax(image)
    coord = np.array(np.unravel_index(max_index, image.shape), np.int32)
    max_value = image[tuple(coord)]
    refined_coord = coord.astype(np.float32)
    dim = coord.size
    for i in range(dim):
        if int(coord[i]) - 1 < 0 or int(coord[i]) + 1 >= image.shape[i]:
            continue
        before_coord = coord.copy()
        before_coord[i] -= 1
        after_coord = coord.copy()
        after_coord[i] += 1
        pa = image[tuple(before_coord)]
        pb = image[tuple(coord)]
        pc = image[tuple(after_coord)]
        diff = 0.5 * (pa - pc) / (pa - 2 * pb + pc)
        refined_coord[i] += diff
    return max_value, refined_coord

def transform_landmarks(landmarks, transformation, output_size=[512,512]):
    transformed_landmarks = copy.deepcopy(landmarks)
    displacement_field = sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat32, size=output_size, outputSpacing=[1,1])
    dim =2
    size = output_size
    spacing = [1]*2
    if dim == 2:
        displacement_field = np.transpose(sitk.GetArrayFromImage(displacement_field), [1, 0, 2])
        mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                           np.array(range(size[1]), np.float32),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=2) * np.expand_dims(np.expand_dims(np.array(spacing, np.float32), axis=0), axis=0)

        for i in range(transformed_landmarks.shape[0]):
            coords = transformed_landmarks[i]
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            distances = np.linalg.norm(vec, axis=2)
            invert_min_distance, transformed_coords = find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            transformed_landmarks[i] = transformed_coords
        return transformed_landmarks


def ElasticDeformation(image, grid_nodes = [5,5], deformation_value=20, image_size=(512, 512)):
    dim = 2
    output_spacing = [1]*dim
    image = sitk.GetImageFromArray(image, isVector=False)
    spline_order = 3
    physical_dimensions = [image_size[i] * output_spacing[i] for i in range(dim)]
    mesh_size = [grid_node - spline_order for grid_node in grid_nodes]
    t = sitk.BSplineTransform(dim, spline_order)
    t.SetTransformDomainOrigin(np.zeros(dim))
    t.SetTransformDomainMeshSize(mesh_size)
    t.SetTransformDomainPhysicalDimensions(physical_dimensions)
    t.SetTransformDomainDirection(np.eye(dim).flatten())
    np.random.seed()
    deform_params = [np.random.uniform(low=-deformation_value, high=deformation_value) for _ in t.GetParameters()]
    t.SetParameters(deform_params)
    return t


def RandomIntensityShiftScale(image, random_shift=[-0.25, 0.25], random_scale=[0.75, 1.25]):
    random.seed()
    random_shift = random.uniform(random_shift[0], random_shift[1])
    random.seed()
    random_scale = random.uniform(random_scale[0], random_scale[1])
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image = sitk.ShiftScale(sitk_image, float(random_shift), float(random_scale))
    return sitk.GetArrayFromImage(sitk_image)

def RandomTranslate(image, translate=20):
    image_size = image.shape
    image = sitk.GetImageFromArray(image)
    current_offset = [np.random.uniform(-translate, translate) for i in range(2)]
    t = sitk.AffineTransform(2)
    t.Translate(current_offset)
    return t

def RandomRotate(image, random_angles=[15]):
    image_size = image.shape
    #center = np.array([ (image_size[0]-1)/2, (image_size[1]-1)/2 ])
    point = [(image_size[i] - 1) * 0.5 for i in range(2)]
    image = sitk.GetImageFromArray(image)
    current_angles = [np.random.uniform(-random_angles[0], random_angles[0])]
    t = sitk.AffineTransform(2)
    t.SetCenter(point)
    t.Rotate(0, 1, angle=current_angles[0])
    return t

def RandomScale(image, scale=[-0.6, 1.4]):
    image = sitk.GetImageFromArray(image)
    t =  sitk.AffineTransform(2)
    scale = random.uniform(scale[0], scale[1])
    t.Scale([scale, scale])
    return t
def Compose(transformations=[]):
    compos = sitk.Transform(2, sitk.sitkIdentity)
    for transformation in transformations:
        compos.AddTransform(transformation)
    return compos
def Input_to_Center(img):
    input_image = sitk.GetImageFromArray(img)
    input_size = input_image.GetSize()
    input_spacing = input_image.GetSpacing()
    input_direction = input_image.GetDirection()
    input_origin = input_image.GetOrigin()
    #print(input_size, input_spacing, input_direction, input_origin)
    dim = 2
    point = [(input_size[i] - 1) * 0.5 for i in range(dim)]
    index = np.matmul(np.matmul(np.diag(1 / np.array(input_spacing)), np.array(input_direction).reshape([dim, dim]).T), (np.array(point) - np.array(input_origin)))
    center = index.tolist()
    t = sitk.AffineTransform(dim)
    t.Translate(center)
    return t

def FitFixedAR(img, output_size=[512,512]):
    input_image = sitk.GetImageFromArray(img)
    input_size, input_spacing = input_image.GetSize(), input_image.GetSpacing()
    current_scale = []
    output_spacing = [1] * 2
    for i in range(2):
        if output_size[i] is None or output_spacing[i] is None:
            continue
        else:
            current_scale.append((input_size[i] * input_spacing[i]) / (output_size[i] * output_spacing[i]))
    max_scale = max(current_scale)
    current_scale = []
    for i in range(2):
        current_scale.append(max_scale)
    s = sitk.AffineTransform(2)
    s.Scale(current_scale)
    return s

def Origin_to_Output_Center(output_size=[512, 512]):
    output_spacing = [1]*2
    output_center = []
    for i in range(2):
        output_center.append((output_size[i] - 1) * output_spacing[i] * 0.5)
    negative_output_center = [-o for o in output_center]
    t = sitk.AffineTransform(2)
    t.Translate(negative_output_center)
    return t

def Resample(img, landmarks,transform, output_size=[512, 512]):
    input_image = sitk.GetImageFromArray(img)
    image_dim = input_image.GetDimension()
    transform_dim = transform.GetDimension()
    assert image_dim == transform_dim, 'image and transform dim must be equal, are ' + str(image_dim) + ' and ' + str(transform_dim)
    output_spacing = [1] * image_dim
    output_origin = [0] * image_dim
    output_direction = np.eye(image_dim).flatten().tolist()
    interpolator = sitk.sitkBSpline
    # resample the image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(output_size)
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetOutputDirection(output_direction)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(input_image.GetPixelID())
    output_image = resample_filter.Execute(input_image)
    landmarks = transform_landmarks(landmarks, resample_filter.GetTransform())
    return sitk.GetArrayFromImage(output_image), landmarks




class Normalize(object):
    def __init__(self, old_range=(0, 1)):
        self.old_range = old_range
    def scale(self, img, old_range, new_range):
        shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
        scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
        return (img + shift) * scale
    def robust_min_max(self, img, consideration_factors=(0.1, 0.1)):
        # sort flattened image
        img_sort = np.sort(img, axis=None)
        # consider x% values
        min_median_index = int(img.size * consideration_factors[0] * 0.5)
        max_median_index = int(img.size * (1 - consideration_factors[1] * 0.5)) - 1
        # return median of highest x% intensity values
        return img_sort[min_median_index], img_sort[max_median_index]
    def __call__(self, sample, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
        #print(torch.is_tensor(sample["image"]))
        if "label" in sample:
            sample["label"] = sample["label"].astype(np.float32)/255.0
        img = sample["image"].astype(np.float)
        min_value, max_value = img.min(), img.max()#self.robust_min_max(img, consideration_factors)
        if max_value == min_value:
            max_value = min_value + 1
        old_range = (min_value, max_value)
        sample["image"] = self.scale(img, old_range, out_range)
        return sample


class ConvertToHeatmaps(object):
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
    def generate_heatmap(self, coords, sigma, device="cpu"):
        # landmark holds the image
        normalize_center = True
        sigma_scale_factor=1.0
        size_sigma_factor = 10
        dim = 2
        heatmap = torch.zeros(self.image_size, dtype=torch.float32, device=device)
        flipped_coords = torch.flip(coords, dims=[0])
        region_start = (flipped_coords - sigma * size_sigma_factor / 2).type(torch.int).to(device)
        region_end = (flipped_coords + sigma * size_sigma_factor / 2).type(torch.int).to(device)

        region_start = torch.max(torch.zeros_like(region_start, device=device), region_start).type(torch.int)
        region_end = torch.min(torch.zeros_like(region_end, device=device).fill_(self.image_size[0]), region_end).type(torch.int)
        
        if torch.any(region_start >= region_end):
            return heatmap

        region_size = (region_end - region_start).type(torch.int)

        sigma = sigma * sigma_scale_factor
        scale = 1

        if normalize_center is not True:
            scale /= math.pow(math.sqrt(2 * math.pi) * sigma, dim)

        dy, dx = torch.meshgrid(torch.tensor(range(region_size[1]), device=device), torch.tensor(range(region_size[0]), device=device))
        x_diff = (dx + region_start[0] - flipped_coords[0])
        y_diff = (dy + region_start[1] - flipped_coords[1])

        squared_distances = (x_diff * x_diff + y_diff * y_diff)
        cropped_heatmap = (scale * torch.exp(-squared_distances / (2 * torch.pow(sigma, 2))))
        heatmap[region_start[0]:region_end[0],
        region_start[1]:region_end[1]] = cropped_heatmap[:, :].T
        
        return heatmap
    
    def __call__(self, landmarks, sigmas):
        device = "cpu"
        if torch.is_tensor(landmarks):
            landmarks = landmarks.squeeze_(dim=0)
            if landmarks.get_device() > -1:
                device = "cuda"
        heatmaps = torch.zeros((landmarks.shape[0], self.image_size[0], self.image_size[1]), dtype=torch.float32, device=device)
        for landmark in range(landmarks.shape[0]):
            heatmaps[landmark] = self.generate_heatmap(landmarks[landmark], sigmas[landmark], device=device)
        heatmaps = heatmaps.unsqueeze_(dim=0)
        return heatmaps

def ToTensor(image, heatmaps):
    """Convert ndarrays in sample to Tensors."""
    #image = np.array(image, dtype=np.float32)
    #image = np.expand_dims(image, axis=1)
    return torch.from_numpy(image), torch.from_numpy(heatmaps)

class ContrastAdjustment(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    def __call__(self, sample):
        img = sample["image"]
        #print(img.dtype, img.shape)
        sample["image"] = self.clahe.apply(sample["image"])
        return sample
'''
class RandomApply(object):
    def __init__(self, transformslist):
        self.transformslist = transformslist
    def __call__(self, sample):
        random.seed()
        random.shuffle(self.transformslist)
        for i in range(len(self.transformslist)):
            sample = self.transformslist[i](sample)
        return sample
class NormalizeBatch(object):
    def __init__(self, feature_wise=False ,train_loader_sample= None, path_to_stats=None):
        self.feature_wise = feature_wise
        self.path_to_stats = path_to_stats
        if(self.feature_wise):
            if(path_to_stats is not None):
                stats = np.loadtxt(path_to_stats)
                mean, stddev = stats[0], stats[1]
            else:
                sample_train = next(iter(train_loader_sample))["image"]
                mean, stddev = sample_train.mean(), sample_train.std()
                stats = np.array([mean, stddev], dtype=np.float32)
                np.savetxt(".\\stats\\batch_stats.txt", stats)
                self.path_to_stats = ".\\stats\\batch_stats.txt"
            self.mean, self.stddev = mean, stddev
    def get_stats_file():
        return str(self.path_to_stats)
    def __call__(self, samplebatch):
        imagebatch = samplebatch["image"]
        if self.feature_wise is False:
            mean, stddev = imagebatch.mean(), imagebatch.std()
        else:
            mean, stddev = self.mean, self.stddev
        imagebatch = (imagebatch-mean)/stddev
        #imagebatch.unsqueeze_(dim=1)
        samplebatch["image"] = imagebatch
        return samplebatch
def argmax(heatmap):
    maxindex = heatmap.argmax()
    return np.flip(np.unravel_index(maxindex, heatmap.shape), axis=0)
class ConvertToHeatmaps(object):
    def __init__(self, num_landmarks=65, sigma=1.5, image_size=(256,256)):
        self.num_landmarks = num_landmarks
        self.sigma = sigma
        self.dim = 2
        self.normalize_center = True
        self.image_size = image_size
        self.size_sigma_factor = 10
        self.scale_factor = 1
    def generate_heatmap(self, coords, sigma_scale_factor=1.0):
        """
        Generates a numpy array of the landmark image for the specified point and parameters.
        :param coords: numpy coordinates ([x], [x, y] or [x, y, z]) of the point.
        :param sigma_scale_factor: Every value of the gaussian is multiplied by this value.
        :return: numpy array of the landmark image.
        """
        # landmark holds the image
        heatmap = np.zeros(self.image_size, dtype=np.float32)
        #print(coords)
        # flip point from [x, y, z] to [z, y, x]
        flipped_coords = np.flip(coords, axis=0)
        region_start = (flipped_coords - self.sigma * self.size_sigma_factor / 2).astype(int)
        region_end = (flipped_coords + self.sigma * self.size_sigma_factor / 2).astype(int)

        region_start = np.maximum(0, region_start).astype(int)
        region_end = np.minimum(self.image_size, region_end).astype(int)

        # return zero landmark, if region is invalid, i.e., landmark is outside of image
        if np.any(region_start >= region_end):
            return heatmap

        region_size = (region_end - region_start).astype(int)

        sigma = self.sigma * sigma_scale_factor
        scale = self.scale_factor

        if not self.normalize_center:
            scale /= math.pow(math.sqrt(2 * math.pi) * sigma, self.dim)

        if self.dim == 2:
            dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]

            squared_distances = x_diff * x_diff + y_diff * y_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0],
                    region_start[1]:region_end[1]] = cropped_heatmap[:, :]
        
        #print(coords, argmax(heatmap))
        return heatmap
    
    def __call__(self, samplebatch):
        landmarksbatch = samplebatch["landmarks"]
        newlandmarksbatch = np.zeros((landmarksbatch.shape[0], landmarksbatch.shape[1], self.image_size[0], self.image_size[1]), dtype=np.float32)
        for batch_idx in range(landmarksbatch.shape[0]):
            for landmark_idx in range(landmarksbatch.shape[1]):
                coords = np.array(landmarksbatch[batch_idx, landmark_idx, :])
                #print(coords.shape)
                newlandmarksbatch[batch_idx, landmark_idx, :,:] = self.generate_heatmap(coords)
                
        samplebatch["landmarks"] = newlandmarksbatch
        return samplebatch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=1)
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
'''