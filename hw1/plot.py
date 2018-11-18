
# coding: utf-8

# In[66]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:

def plot(x, y, std=None,title=None, xlable=None, ylable=None, fontsize=20, linestyle='-', color='r'):
    fig = plt.figure()
    plt.xlabel(xlable, fontsize=fontsize)
    plt.ylabel(ylable, fontsize=fontsize)
    plt.title(title, fontsize=fontsize+5)
    plt.legend(loc='bottom left')
    
    sns.tsplot(time=x, data=y, color=color, linestyle=linestyle)
    
    plt.xlim([np.min(x)-.5, np.max(x)+.5])
    #plt.ylim([np.min(y)-.5, np.max(y)+.5])
    
    if std is not None:
        plt.errorbar(x=x, y=y, yerr=std, fmt='o')
    
    plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



