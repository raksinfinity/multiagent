import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy.signal as ss
#run all nbot count
lst=[1,2,3,4,5,6,7,8,9,0]
iters=[1e5,1e5,1e5,1e5,2e5,2e5,2e5,.1e5,.1e5]
#nbot from 2 to 10
for nbot in range(2,11):
  #hyper parameters
  ep=.1
  lam=.9
  dep=0
  it=int(iters[nbot-2])
  on=[12,13,14,15,16,17,18,19,20,21]
  off=[5,5,5,5,5,7,7,7,7,7]
  ofpwr=np.array(off)[:nbot]
  onpwr=np.array(on)[:nbot]
  fulp=np.array([220,196])-np.sum(on[nbot:])-np.sum(off[nbot:])

  #create space state
  voc='0123456789'[:nbot]
  res=[]
  def arr(st):
    if len(st)==nbot: res.append(st)
    rest=[s for s in voc if st.find(s)==-1]
    for i in rest:
      arr(st+i)
  arr('')    
  spaln=len(res)
  actln=2**nbot

  #reward function
  def reward(st0,act,dep,prin):
    st=res[st0]
    act=bin(act)[2:]
    act='0'*(nbot-len(act))+act
    #ofs=[i for i in range(len(act)) if act[i]=='0']
    ons=[str(i) for i in range(len(act)) if act[i]=='1']  
    st1=st[:]
    inds=[]
    for i in ons:
      inds.append(st1.index(i))
    am=np.argsort(inds)
    ons=list(np.array(ons)[am])
    onind=[int(j) for j in ons]  
    for i in ons:  
      ind=st1.index(i)
      st1=st1[:ind]+st1[ind+1:]+st1[ind]
    pwr=np.sum(onpwr[onind])+np.sum(ofpwr[onind])
    if pwr<fulp[dep]:
      rew=int(np.sum([(nbot-st.index(i))*3 for i in ons])*pwr/fulp[dep]) #*pwr/fulp[dep] 
    else: rew=0
    st1=''.join(st1)
    if prin: print(st,st1,ons,rew)
    st1=res.index(st1)
    return rew,st1,ons

  #train
  st=0
  sts=[]; acts=[]; onrobs=[]
  qtab=np.ones((spaln,actln),dtype='uint8')
  qplu=[]
  for i in range(actln):
    bn=bin(i)[2:]
    oncn=len([j for j in bn if j=='1'])
    if oncn>nbot//2: qplu.append(i)
  #qtab[:,qplu]=2
  for i in range(it):
    if not i%(it//20): print(int(i/it*100),end=' ')
    mx=np.argmax(qtab[st,:])
    act=np.random.choice([mx,0],p=[ep,1-ep])
    if act==0: act=np.random.choice(actln)
    rew,st1,ons=reward(st,act,dep,prin=0)
    qtab[st,act]=qtab[st,act]*(1-lam) + lam*rew
    st=st1
    sts.append(st)
    acts.append(act)
    onrobs.append(len(ons))
  print('')

  #learning proc
  win=200
  ln=len(onrobs)
  smo=np.zeros(ln-win)
  for i in range(ln-win):
    smo[i]=np.mean(onrobs[i:i+win])

  #play
  st=0
  sts=[]
  acts=[]
  onrun=[]
  for i in range(100):
    act=np.argmax(qtab[st,:])
    rew,st1,ons=reward(st,act,dep,prin=0)
    st=st1
    sts.append(st)
    acts.append(act)
    onrun.append(len(ons))

  lst[nbot-1]=[onrun,smo]

#forming plots
#get learning curves by nbot count
xl=100
sigs=np.zeros((9,xl))
xs=np.zeros((9,xl))
ns=np.zeros((9,xl))
stds=np.zeros(9)
spr=np.zeros(9)
der=np.zeros(9)
derstd=np.zeros(9)
for i in range(1,10):
  s=lst[i][1]
  s=ss.resample(s,xl)
  cur=s
  cur=-abs(i-cur)+i
  cur=cur/np.max([1,i])
  stds[i-1]=np.std(cur)
  spr[i-1]=np.max(cur)-np.min(cur)
  der[i-1]=np.mean(abs(np.diff(cur)))
  derstd[i-1]=np.std(abs(np.diff(cur)))
  sigs[i-1,:]=cur
  xs[i-1,:]=np.arange(xl)
  ns[i-1,:]=i+1
sre=np.ndarray.flatten(sigs)**.9
xre=np.ndarray.flatten(xs)
nre=np.ndarray.flatten(ns)
#plot statistics
plt.figure()
plt.plot(stds)
plt.title('Standard deviation of learning curve')
plt.xlabel('bots number')
plt.ylabel('std')
plt.figure()
plt.plot(spr)
plt.title('Average spread of learning curve')
plt.xlabel('bots number')
plt.ylabel('max - min')
plt.figure()
plt.plot(der)
plt.title('Average derivative of learning curve')
plt.xlabel('bots number')
plt.ylabel('mean derivative')
plt.figure()
plt.plot(derstd)
plt.title('Standard deviation of derivative of learning curve')
plt.xlabel('bots number')
#draw distribution
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure(figsize=(20,10))
ax = Axes3D(fig)
surf = ax.plot_trisurf( nre, xre,sre, cmap=cm.viridis, alpha=0.8, linewidth=0, edgecolor='none')
ax.set_xlabel('number of bots')
ax.set_zlabel('portion of cooperative')
ax.set_ylabel('time in %')
ax.set_title('learning curves')
cb1=fig.colorbar(surf, shrink=.8, aspect=20) 
cb1.ax.set_ylabel('Q-learning', labelpad=10, rotation=270)
ax.view_init(20, 50)
