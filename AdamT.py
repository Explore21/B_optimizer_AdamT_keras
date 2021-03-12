from keras import backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from tensorflow.python.ops import math_ops

class AdamT(Optimizer):
 

    def __init__(self, learning_rate=0.001, beta_1=0.9,beta_2=0.999, beta_3=0.9,beta_4=0.999,gamma_1=0.9, gamma_2=0.999, phi_1=0.5, phi_2=0.5, amsgrad=False, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(AdamT, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.beta_3 = K.variable(beta_3, name='beta_3')
            self.beta_4 = K.variable(beta_4, name='beta_4')
            self.gamma_1 = K.variable(gamma_1, name='gamma_1')
            self.gamma_2 = K.variable(gamma_2, name='gamma_2')
            self.phi_1 = K.variable(phi_1, name='phi_1')
            self.phi_2 = K.variable(phi_2, name='phi_2')
            self.decay = K.variable(self.initial_decay, name='decay')
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        lms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='lm_' + str(i))
              for (i, p) in enumerate(params)]
        lvs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='lv_' + str(i))
              for (i, p) in enumerate(params)]
        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]

        bms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='bm_' + str(i))
              for (i, p) in enumerate(params)]
        
        bvs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='bv_' + str(i))
              for (i, p) in enumerate(params)]
        
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                     dtype=K.dtype(p),
                     name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats
 

        for p, g, m, v, lm, lv, bm, bv, vhat in zip(params, grads, ms,vs,lms, lvs, bms, bvs, vhats):
            
       
            lm_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            
            lv_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            
            bm_t = (self.gamma_1*self.phi_1)*bm + (1-self.gamma_1)*(lm_t-lm)
            bv_t = (self.gamma_2*self.phi_2)*bv + (1-self.gamma_2)*(lv_t-lv)
            
            
            #f1=(1-self.gamma_1*self.phi_1)/ (1-self.gamma_1)*(1- (self.gamma_1*self.phi_1)**t) is approximately 5
            #f2= ((1-self.gamma_2*self.phi_2)*bv_t)/((1-self.gamma_2)*(1-(self.gamma_2*self.phi_2)**t)) is approximately 5
            
            m_t= lm_t +((1-self.gamma_1*self.phi_1)*bm_t*5 ) 
  
            
            v_t= lv_t  +((1-self.gamma_2*self.phi_2)*bv_t*5 ) 
           
            
 
            
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            
      
            

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(lv, lv_t))
            self.updates.append(K.update(lm, lm_t))
            self.updates.append(K.update(bv, bv_t))
            self.updates.append(K.update(bm, bm_t))
            new_p = p_t
 
 

            
            
            
            
            
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
