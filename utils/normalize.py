import torchvision.transforms as transforms

def get_normalizer(data_type):
    """
        Get a transformation that normalizes the input data to have zero mean and unit variance
        
        Parameters
        ----------
            - data_type: str: can be mfcc, mfcc-delta, logfbank_40, logfbank_40-delta

        Return
        ------
            - normalize: torchvision.transforms
    """
    normalize = None

    if data_type == 'mfcc':
        normalize = transforms.Normalize(mean=[-4.9782], std=[15.7283])
    elif data_type == 'mfcc-delta':
        normalize = transforms.Normalize(mean=[-4.9783,  0.0205, -0.0229],
                                         std=[15.7283,  4.2711,  3.3977])
    elif data_type == 'ssc':
        normalize = transforms.Normalize(mean=[2.4145e+03], std=[2090.6826])
    elif data_type == 'ssc-delta':
        normalize = transforms.Normalize(mean=[2.4145e+03, -6.5548e-02,  4.1564e-03],
                                         std=[2090.6826, 24.6444, 19.4945])
    elif data_type == 'logfbank':
        normalize = transforms.Normalize(mean=[8.2591], std=[3.6642])
    elif data_type == 'logfbank-delta':
        normalize = transforms.Normalize(mean=[8.2591, 0.0217, 0.0067],
                                         std=[3.6642, 0.8152, 0.5713])
    elif data_type == 'logfbank_40':
        normalize = transforms.Normalize(mean=[7.7120], std=[3.7074])
    elif data_type == 'logfbank_40-delta':
        normalize = transforms.Normalize(mean=[7.7120, 0.0359, 0.0084],
                                         std=[3.7074, 0.8747, 0.6347])

    return normalize

if __name__ == '__main__':
    pass