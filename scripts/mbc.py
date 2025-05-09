import pinocchio as pin
import numpy as np
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
import matplotlib.pylab as plt; plt.ion()
from rich import print
from tabulate import tabulate
import mujoco
# import mujoco.viewer
# from rich import print as rprint

class MBC:
    def __init__(self, modelfilename_pin):
        # 从MJCF文件构建机器人模型，并使用自由飞行根关节
        self.model = pin.buildModelFromMJCF(modelfilename_pin, root_joint=pin.JointModelFreeFlyer())
        # 创建Pinocchio数据
        self.data = self.model.createData()
        # 初始化机器人位置，使用模型的中性姿态
        self.q = pin.neutral(self.model).copy()
        self.q[2] = 0.8  # 设置Z轴高度为1.10
        # 初始化机器人速度为零
        self.qdot = np.zeros(self.model.nv)
        # 构建传感器名称列表，左右两边的传感器名称拼接
        self.fclist = ['{}_sensor_{}{}'.format('l', i, j) for i in range(1, 4) for j in range(1, 5)] + \
                      ['{}_sensor_{}{}'.format('r', i, j) for i in range(1, 4) for j in range(1, 5)]
        # 初始化接触列表为空
        self.clist = []
        # 获取传感器数量
        self.fcN = len(self.fclist)
        self.cN = len(self.clist)
        # 自由任务下的加速度参考初始化为零
        self.aqfree = np.zeros(self.model.nv)
        # 动力学计算相关的变量初始化
        self.M, self.c, self.com, self.Jcom, self.vcom = None, None, None, None, None
        self.com_task_qobj = None
        # 接触信息的初始化
        self.contactposition = np.zeros(3 * self.cN)
        self.contactvelocity = np.zeros(3 * self.cN)
        self.Jdv = np.zeros(3 * self.cN)
        self.Jdvc = np.zeros(3 * self.cN)
        self.o_Jtool = np.zeros((3 * self.cN, self.model.nv))
        self.eR = 0.01
        self.Gde = None

        # 关节控制权重矩阵，排除自由度（前六个被认为是漂移自由度）
        self.Wtau = np.diag([0.0008] * (self.model.nv - 6))
        # 任务及加速度任务相关变量初始化
        self.jW = None
        self.jWe = 10
        self.jWdiag = None
        self.jlist = None
        self.jN = None
        self.qobj = None
        self.kpj = 100  # 位置控制增益
        self.kdj = 10   # 速度控制增益
        self.ajW = None
        self.ajWe = 10
        self.ajlist = None
        self.ajN = None
        self.aqobj = None

        # 用于足部和臀部相关的变量初始化
        self.rfpos, self.lfpos, self.rhip, self.lhip, self.fRT, self.fpi = None, None, None, None, None, None
        
        # 脚的状态初始化，0代表某一种状态（如落地）
        self.fs = 0
        # 接触计数初始化，对应每个传感器的接触力度
        self.concou = np.zeros(len(self.fclist))
        # 机器人步态参数初始化：膝盖抬高角度、步态幅度、踝关节偏置及偏移因子
        self.kneeliftangle = 1.0  # 膝盖抬升角度
        self.amoo = 0.08         # 步态振幅参数
        self.anklebias = 0.05    # 踝关节偏置参数
        self.b = 0.3             # 其他步态调节参数
        
        self.mvcom = None
        self.centerofhip = None
        self.lramp = 0.2
        self.hipzl = 0.0 
        self.hipzr = 0.0

    def getDK(self):
        model = self.model
        data = self.data
        self.M = pin.crba(model, data, self.q)
        pin.computeMinverse(model,data,self.q)
        self.c = pin.nle(model, data, self.q, self.qdot)
        pin.forwardKinematics(model,data,self.q,self.qdot,np.zeros(model.nv))
        pin.updateFramePlacements(model, data)
        com = pin.centerOfMass(model,data,self.q,False)
        self.com = com
        self.Jcom = pin.jacobianCenterOfMass(model,data,self.q,False)
        self.vcom = self.Jcom@self.qdot
        rfid = model.getFrameId('r_leg_ankle_pitch_Link')
        lfid = model.getFrameId('l_leg_ankle_pitch_Link')
        rhid = model.getFrameId('r_leg_hip_roll_Link')
        lhid = model.getFrameId('l_leg_hip_roll_Link')
        self.rfpos = data.oMf[rfid].translation
        self.lfpos = data.oMf[lfid].translation
        self.rhip = data.oMf[rhid].translation
        self.lhip = data.oMf[lhid].translation
        vecr2l = self.lhip - self.rhip
        vecr2lxynorm = np.sqrt(vecr2l[0]*vecr2l[0] + vecr2l[1]*vecr2l[1])
        fRy = np.array([vecr2l[0]/vecr2lxynorm,vecr2l[1]/vecr2lxynorm,0])
        fRz = np.array([0,0,1])
        fRx = np.cross(fRy,fRz)
        self.fRT = np.array([fRx,fRy,fRz])
        # fp = vecr2l/2 + self.rhip
        fp = self.centerofhip
        self.fpi = -self.fRT@fp
        
    def getCon(self):
        model = self.model
        data = self.data
        self.cN = len(self.clist)
        self.contactposition = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.contactposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(self.clist[i])].translation
        self.contactvelocity = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.contactvelocity[3*i:3*(i+1)] = pin.getFrameVelocity(model, data, model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED).linear
        self.Jdv = np.zeros(3*self.cN)
        self.Jdvc = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.Jdv[3*i:3*(i+1)] = pin.getFrameAcceleration(model,data,model.getFrameId(self.clist[i]),pin.LOCAL_WORLD_ALIGNED).linear
            self.Jdvc[3*i:3*(i+1)] = pin.getFrameClassicalAcceleration(model, data, model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED).linear
        self.o_Jtool = np.zeros((3*self.cN,model.nv))
        for i in range(self.cN):
            self.o_Jtool[3*i:3*(i+1),:] = pin.computeFrameJacobian(model,data,self.q,model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED)[:3,:]
        self.Gde = self.o_Jtool@data.Minv@self.o_Jtool.T + self.eR*np.identity(3*self.cN)
    

    def tsid(self):
        model = self.model
        # self.getDK()
        self.getCon()
        if self.jlist is None:
            self.jN = 0
        else:
            self.jN = len(self.jlist)
        self.jW = np.diag([self.jWe]*self.jN)
        if self.jWdiag:
            self.jW = np.diag(self.jWdiag)
        # ratoajw = 1
        # if self.jN>10:
        #     self.jW[4][4]=ratoajw*self.jWe
        #     self.jW[5][5]=ratoajw*self.jWe
        #     self.jW[10][10]=ratoajw*self.jWe
        #     self.jW[11][11]=ratoajw*self.jWe
        if self.ajlist is None:
            self.ajN = 0
        else:
            self.ajN = len(self.ajlist)
        self.ajW = np.diag([self.ajWe]*self.ajN)
        while True:
            self.getCon()
            Pqdd = np.zeros((model.nv+3*self.cN,model.nv+3*self.cN))
            qqdd = np.zeros(model.nv+3*self.cN)
            Jes = np.block([self.M[6:,:],-self.o_Jtool.T[6:,:]])
            esbias = -self.c[6:]
            Pqdd += Jes.T@self.Wtau@Jes
            qqdd += -Jes.T@self.Wtau@esbias
            if self.jN != 0:
                Jj = np.zeros((self.jN,model.nv+3*self.cN))
                for j in range(self.jN):
                    Jj[j,self.jlist[j]] = 1
                qhere = np.array([self.q[i+1] for i in self.jlist])
                qdhere = np.array([self.qdot[i] for i in self.jlist])
                task_ja = -self.kpj*(qhere - self.qobj) - self.kdj*qdhere
                Pqdd += Jj.T@self.jW@Jj
                qqdd += -(Jj.T)@self.jW@task_ja
            if self.ajN != 0:
                aJj = np.zeros((self.ajN,model.nv+3*self.cN))
                for j in range(self.ajN):
                    aJj[j,self.ajlist[j]] = 1
                task_aja = self.aqobj
                Pqdd += aJj.T@self.ajW@aJj
                qqdd += -(aJj.T)@self.ajW@task_aja
            
            QPP  = Pqdd + 0.0001*np.identity(model.nv+3*self.cN)
            QPq = qqdd
            A = np.block([[self.M[:6,:],-self.o_Jtool.T[:6,:]],[self.o_Jtool,0.001*np.identity(3*self.cN)]])
            A[:6,:6] += 0.00001*np.identity(6)
            b = np.concatenate((-self.c[:6],-self.Jdv - 5*(self.contactvelocity)))#- 20*(contactposition - origincontactposition) - 5*(contactvelocity) -self.Jdv
            maxset = 100000
            lb = np.array([-maxset]*6+[-15000]*12+[-19000,-19000,-19000]*(self.cN)) #10.1
            ub = np.array([ maxset]*6+[ 15000]*12+[ 19000]*(3*self.cN))
            # self.x = solve_qp(QPP,QPq,A=A,b=b,lb=lb,ub=ub,solver="proxqp").copy()
            self.x = proxqp_solve_qp(QPP,QPq,A=A,b=b,lb=lb,ub=ub,eps_abs=1e-7)
            liftoffstate = 0
            herelist = self.clist.copy()
            for i in range(self.cN):
                if self.clist  and self.x[model.nv+2+3*i] <= -0.00001: #投影到法向
                    herelist.remove(self.clist[i])
                    liftoffstate += 1
            self.clist = herelist
            if liftoffstate == 0:
                break
        return self.M[6:,:]@self.x[:model.nv] + (self.c[6:]) - self.o_Jtool.T[6:,:]@self.x[model.nv:]


def walk_controller(mbc: MBC, ppos, qvel, concou, time=0):
    """
    Walk controller computes the joint accelerations for walking behavior.
    
    Parameters:
        mbc   : MBC object representing the robot model.
        ppos : Array of joint positions.
        qvel : Array of joint velocities.
        time : Simulation time (default 0).
    
    Returns:
        The computed task-space acceleration from tsid().
    """
    # Initialize leg contact flags and update kinematics
    rls, lls = 0, 0
    mbc.getDK()
    # Update the model's joint states and sensor contact list
    mbc.q = ppos
    mbc.qdot = qvel
    mbc.clist = [fc for fc, cnt in zip(mbc.fclist, concou) if cnt]

    mbc.vcom = mbc.mvcom

    # Physical constants and center-of-mass dynamics
    h = 1
    g = 9.8
    omega = np.sqrt(g / h)  # Natural frequency of the pendulum
    eta = mbc.com + mbc.vcom / omega  # Extrapolated CoM

    # Transform positions to foot frame reference
    frfpos = mbc.fRT @ mbc.rfpos + mbc.fpi
    flfpos = mbc.fRT @ mbc.lfpos + mbc.fpi
    fcom = mbc.fRT @ mbc.com + mbc.fpi
    fvcom = mbc.fRT @ mbc.vcom
    fvp = mbc.fRT @ (mbc.vcom / omega)
    feta = fcom + fvp

    # Determine contact status (using threshold on contact force)
    lls = int(np.any(concou[:int(len(mbc.fclist)/2)]))
    rls = int(np.any(concou[int(len(mbc.fclist)/2):]))

    # Update stance phase state based on vertical displacement conditions
    if mbc.fs == 0 and (feta[1] - frfpos[1]) > mbc.lramp:
        print(f"time: {time}, fs_state_transition: 0 -> 1")
        mbc.fs = 1
    elif mbc.fs == 1 and (flfpos[1] - feta[1]) > mbc.lramp:
        print(f"time: {time}, fs_state_transition: 1 -> 0")
        mbc.fs = 0

    # Initialize control output and joint target variables
    output, hipyor, hipyol, apitr, apitl = 0, 0, 0, 0, 0
    kneer, kneel = 0, 0
    hipzl , hipzr = 0,0

    # Configurable gait parameters
    kneeliftangle = mbc.kneeliftangle  # Knee lifting angle
    amoo = mbc.amoo                   # Step amplitude parameter
    anklebias = mbc.anklebias         # Ankle pitch offset
    b = mbc.b                         # Additional gait modulation parameter

    # Set output force based on stance phase
    output = amoo if mbc.fs == 1 else -amoo

    # Left stance phase when fs is 0
    if mbc.fs == 0:
        kneel = kneeliftangle
        diffpos = fcom[0] + b * fvcom[0] - frfpos[0]
        ang = diffpos

        if lls == 0:
            if abs(diffpos) >= 2 * 0.55:
                print("error!!!!!!")
            ang = np.arcsin(diffpos / (2 * 0.55))
            hipyor = ang
            hipyol = -ang
            hipzr = mbc.hipzr#0.2

        hipyol -= (kneeliftangle / 2)
        apitl += -anklebias - (kneeliftangle / 2)
        apitr += -anklebias

        if fvcom[1] > 0:
            kneel = 0
            hipyol += kneeliftangle / 2
            apitl += kneeliftangle / 2

    # Right stance phase when fs is 1
    elif mbc.fs == 1:
        kneer = kneeliftangle
        diffpos = fcom[0] + b * fvcom[0] - flfpos[0]
        ang = diffpos

        if rls == 0:
            if abs(diffpos) >= 2 * 0.55:
                print("error!!!!!!")
            ang = np.arcsin(diffpos / (2 * 0.55))
            hipyor = -ang
            hipyol = ang
            hipzl = mbc.hipzl#-0.2

        hipyor -= (kneeliftangle / 2)
        apitr += -anklebias - (kneeliftangle / 2)
        apitl += -anklebias

        if fvcom[1] < 0:
            kneer = 0
            hipyor += kneeliftangle / 2
            apitr += kneeliftangle / 2



    # Define joint task indices and corresponding target joint angles
    # [hip pitch, hip yaw, hip roll, knee, ankle pitch, ankle roll] for left and right legs
    mbc.jlist = [
        6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17,
    ]
    mbc.qobj = [
        output, hipzl, hipyol, kneel, apitl, 0,
        output, hipzr, hipyor, kneer, apitr, 0,
    ]
    # return np.array(mbc.qobj) * 180 / np.pi
    return mbc.tsid()
