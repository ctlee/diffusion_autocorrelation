#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


static PyObject* genTraj(PyObject* self, PyObject* args)
{
    PyArrayObject *positions, *velocities, *noises, *forces;
    double k, m, c, a, b, pos, dt;
    int N;
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddddddi", 
                &PyArray_Type, &positions,      
                &PyArray_Type, &velocities,
                &PyArray_Type, &noises,
                &PyArray_Type, &forces,
                &k, &m, &c, &a, &b, &pos, &dt, &N))
        return NULL;
    double vel = 0;
    double force = 0;
    int i; 
    double bdt = b*dt;
    double mm = 2*m;
    double bdtmm = bdt/mm;
    for (i = 0; i < N; i++) {
        *((double *) PyArray_GETPTR1(positions, i)) = pos;
        double bump = *((double *) PyArray_GETPTR1(noises, i));
        pos = pos + bdt*vel + bdtmm*dt*force + bdtmm*bump;
        double fnew = -k*pos;
        *((double *) PyArray_GETPTR1(forces, i)) = fnew;
        vel = a*vel + dt/(mm)*(a*force + fnew) +  b/m*bump;
        force = fnew;
        *((double *) PyArray_GETPTR1(velocities, i)) = vel;
    }
    return Py_BuildValue("i", 1);
}


/*  In place calculation of sample langevin trajectory just for the heck of it
 *  TODO add bias 
 */
static PyObject* generateTrajectory(PyObject* self, PyObject* args)
{
    PyArrayObject *positions, *velocities, *noises, *forces, *ke, *pe;
    double k, m, c, a, b, pos, dt;
    int N;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!dddddddi", 
                &PyArray_Type, &positions,      
                &PyArray_Type, &velocities,
                &PyArray_Type, &noises,
                &PyArray_Type, &forces,
                &PyArray_Type, &ke,
                &PyArray_Type, &pe,
                &k, &m, &c, &a, &b, &pos, &dt, &N))
        return NULL;

    double vel = 0;
    double force = 0;
    int i;
    double bdt = b*dt;
    double mm = 2*m;
    double bdtmm = bdt/mm;
    // LOOP
    for (i = 0; i < N; i++) {
        *((double *) PyArray_GETPTR1(positions, i)) = pos;
        double bump = *((double *) PyArray_GETPTR1(noises, i));
        pos = pos + bdt*vel + bdtmm*dt*force + bdtmm*bump;
        double fnew = -k*pos;
        *((double *) PyArray_GETPTR1(forces, i)) = fnew;
        vel = a*vel + dt/(mm)*(a*force + fnew) +  b/m*bump;
        force = fnew;
        *((double *) PyArray_GETPTR1(velocities, i)) = vel;
        *((double *) PyArray_GETPTR1(ke, i)) = 0.5*m*vel*vel;
        *((double *) PyArray_GETPTR1(pe, i)) = 0.5*k*pos*pos;
    }
    return Py_BuildValue("i", 1);
}

/*  define functions in module */
static PyMethodDef TrajTools[] =
{
    {"genTraj", genTraj, METH_VARARGS, "generate trajectory"},
    {"generateTrajectory", generateTrajectory, METH_VARARGS, "Create a sample trajectory"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
inittraj_tools(void)
{
    (void) Py_InitModule("traj_tools", TrajTools);
    import_array();
}
