#include "mjpc/simulate.h"  // mjpc fork

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>

// 添加正确的OpenGL头文件包含
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "lodepng.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <platform_ui_adapter.h>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/utilities.h"

// 包含仪表盘头文件
#include "mjpc/tasks/simple_car/dashboard.h"

#ifdef __APPLE__
std::string GetSavePath(const char* filename);
#else
static std::string GetSavePath(const char* filename) {
  return filename;
}
#endif

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

using Seconds = std::chrono::duration<double>;
using Milliseconds = std::chrono::duration<double, std::milli>;

//------------------------------------------- global -----------------------------------------------

const int maxgeom = 5000;            // preallocated geom array in mjvScene
const double zoom_increment = 0.02;  // ratio of one click-wheel zoom increment to vertical extent

// section ids
enum {
  // left ui
  SECT_FILE = 0,
  SECT_OPTION,
  SECT_SIMULATION,
  SECT_WATCH,
  SECT_TASK,
  SECT_AGENT,
  SECT_ESTIMATOR,
  SECT_PHYSICS,
  SECT_RENDERING,
  SECT_GROUP,
  NSECT0,

  // right ui
  SECT_JOINT = 0,
  SECT_CONTROL,
  NSECT1
};

// file section of UI
const mjuiDef defFile[] = {
  {mjITEM_SECTION,   "File",          0, nullptr,                    "AF"},
  {mjITEM_BUTTON,    "Save xml",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Save mjb",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Print model",   2, nullptr,                    "CM"},
  {mjITEM_BUTTON,    "Print data",    2, nullptr,                    "CD"},
  {mjITEM_BUTTON,    "Quit",          1, nullptr,                    "CQ"},
  {mjITEM_BUTTON,    "Screenshot",    2, nullptr,                    "CP"},
  {mjITEM_END}
};

// help strings
const char help_content[] =
  "Space\n"
  "+  -\n"
  "Right arrow\n"
  "[  ]\n"
  "Esc\n"
  "Double-click\n"
  "Page Up\n"
  "Right double-click\n"
  "Ctrl Right double-click\n"
  "Scroll, middle drag\n"
  "Left drag\n"
  "[Shift] right drag\n"
  "Ctrl [Shift] drag\n"
  "Ctrl [Shift] right drag\n"
  "F1\n"
  "F2\n"
  "F3\n"
  "F4\n"
  "F5\n"
  "UI right hold\n"
  "UI title double-click";

const char help_title[] =
  "Play / Pause\n"
  "Speed up / down\n"
  "Step\n"
  "Cycle cameras\n"
  "Free camera\n"
  "Select\n"
  "Select parent\n"
  "Center\n"
  "Tracking camera\n"
  "Zoom\n"
  "View rotate\n"
  "View translate\n"
  "Object rotate\n"
  "Object translate\n"
  "Help\n"
  "Info\n"
  "Profiler\n"
  "Sensors\n"
  "Full screen\n"
  "Show UI shortcuts\n"
  "Expand/collapse all";


//-------------------------------- profiler, sensor, info, watch -----------------------------------

// number of lines in the Constraint ("Counts") and Cost ("Convergence") figures
static constexpr int kConstraintNum = 5;
static constexpr int kCostNum = 3;

// init profiler figures
void InitializeProfiler(mj::Simulate* sim) {
  // set figures to default
  mjv_defaultFigure(&sim->figconstraint);
  mjv_defaultFigure(&sim->figcost);
  mjv_defaultFigure(&sim->figtimer);
  mjv_defaultFigure(&sim->figsize);

  // titles
  mju::strcpy_arr(sim->figconstraint.title, "Counts");
  mju::strcpy_arr(sim->figcost.title, "Convergence (log 10)");
  mju::strcpy_arr(sim->figsize.title, "Dimensions");
  mju::strcpy_arr(sim->figtimer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(sim->figconstraint.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figcost.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figsize.xlabel, "Video frame");
  mju::strcpy_arr(sim->figtimer.xlabel, "Video frame");

  // y-tick number formats
  mju::strcpy_arr(sim->figconstraint.yformat, "%.0f");
  mju::strcpy_arr(sim->figcost.yformat, "%.1f");
  mju::strcpy_arr(sim->figsize.yformat, "%.0f");
  mju::strcpy_arr(sim->figtimer.yformat, "%.2f");

  // colors
  sim->figconstraint.figurergba[0] = 0.1f;
  sim->figcost.figurergba[2]       = 0.2f;
  sim->figsize.figurergba[0]       = 0.1f;
  sim->figtimer.figurergba[2]      = 0.2f;
  sim->figconstraint.figurergba[3] = 0.5f;
  sim->figcost.figurergba[3]       = 0.5f;
  sim->figsize.figurergba[3]       = 0.5f;
  sim->figtimer.figurergba[3]      = 0.5f;

  // repeat line colors for constraint and cost figures
  mjvFigure* fig = &sim->figcost;
  for (int i=kCostNum; i<mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kCostNum][0];
    fig->linergb[i][1] = fig->linergb[i - kCostNum][1];
    fig->linergb[i][2] = fig->linergb[i - kCostNum][2];
  }
  fig = &sim->figconstraint;
  for (int i=kConstraintNum; i<mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kConstraintNum][0];
    fig->linergb[i][1] = fig->linergb[i - kConstraintNum][1];
    fig->linergb[i][2] = fig->linergb[i - kConstraintNum][2];
  }

  // legends
  mju::strcpy_arr(sim->figconstraint.linename[0], "total");
  mju::strcpy_arr(sim->figconstraint.linename[1], "active");
  mju::strcpy_arr(sim->figconstraint.linename[2], "changed");
  mju::strcpy_arr(sim->figconstraint.linename[3], "evals");
  mju::strcpy_arr(sim->figconstraint.linename[4], "updates");
  mju::strcpy_arr(sim->figcost.linename[0], "improvement");
  mju::strcpy_arr(sim->figcost.linename[1], "gradient");
  mju::strcpy_arr(sim->figcost.linename[2], "lineslope");
  mju::strcpy_arr(sim->figsize.linename[0], "dof");
  mju::strcpy_arr(sim->figsize.linename[1], "body");
  mju::strcpy_arr(sim->figsize.linename[2], "constraint");
  mju::strcpy_arr(sim->figsize.linename[3], "sqrt(nnz)");
  mju::strcpy_arr(sim->figsize.linename[4], "contact");
  mju::strcpy_arr(sim->figsize.linename[5], "iteration");
  mju::strcpy_arr(sim->figtimer.linename[0], "total");
  mju::strcpy_arr(sim->figtimer.linename[1], "collision");
  mju::strcpy_arr(sim->figtimer.linename[2], "prepare");
  mju::strcpy_arr(sim->figtimer.linename[3], "solve");
  mju::strcpy_arr(sim->figtimer.linename[4], "other");

  // grid sizes
  sim->figconstraint.gridsize[0] = 5;
  sim->figconstraint.gridsize[1] = 5;
  sim->figcost.gridsize[0] = 5;
  sim->figcost.gridsize[1] = 5;
  sim->figsize.gridsize[0] = 3;
  sim->figsize.gridsize[1] = 5;
  sim->figtimer.gridsize[0] = 3;
  sim->figtimer.gridsize[1] = 5;

  // minimum ranges
  sim->figconstraint.range[0][0] = 0;
  sim->figconstraint.range[0][1] = 20;
  sim->figconstraint.range[1][0] = 0;
  sim->figconstraint.range[1][1] = 80;
  sim->figcost.range[0][0] = 0;
  sim->figcost.range[0][1] = 20;
  sim->figcost.range[1][0] = -15;
  sim->figcost.range[1][1] = 5;
  sim->figsize.range[0][0] = -200;
  sim->figsize.range[0][1] = 0;
  sim->figsize.range[1][0] = 0;
  sim->figsize.range[1][1] = 100;
  sim->figtimer.range[0][0] = -200;
  sim->figtimer.range[0][1] = 0;
  sim->figtimer.range[1][0] = 0;
  sim->figtimer.range[1][1] = 0.4f;

  // init x axis on history figures (do not show yet)
  for (int n=0; n<6; n++) {
    for (int i=0; i<mjMAXLINEPNT; i++) {
      sim->figtimer.linedata[n][2*i] = -i;
      sim->figsize.linedata[n][2*i] = -i;
    }
  }
}

// update profiler figures
void UpdateProfiler(mj::Simulate* sim) {
  // reset lines in Constraint and Cost figures
  memset(sim->figconstraint.linepnt, 0, mjMAXLINE*sizeof(int));
  memset(sim->figcost.linepnt, 0, mjMAXLINE*sizeof(int));

  // number of islands that have diagnostics
  int nisland = mjMIN(sim->d->solver_nisland, mjNISLAND);

  // iterate over islands
  for (int k=0; k < nisland; k++) {
    // ==== update Constraint ("Counts") figure

    // number of points to plot, starting line
    int npoints = mjMIN(mjMIN(sim->d->solver_niter[k], mjNSOLVER), mjMAXLINEPNT);
    int start = kConstraintNum * k;

    sim->figconstraint.linepnt[start + 0] = npoints;
    for (int i=1; i < kConstraintNum; i++) {
      sim->figconstraint.linepnt[start + i] = npoints;
    }
    if (sim->m->opt.solver == mjSOL_PGS) {
      sim->figconstraint.linepnt[start + 3] = 0;
      sim->figconstraint.linepnt[start + 4] = 0;
    }
    if (sim->m->opt.solver == mjSOL_CG) {
      sim->figconstraint.linepnt[start + 4] = 0;
    }
    for (int i=0; i<npoints; i++) {
      // x
      sim->figconstraint.linedata[start + 0][2*i] = i;
      sim->figconstraint.linedata[start + 1][2*i] = i;
      sim->figconstraint.linedata[start + 2][2*i] = i;
      sim->figconstraint.linedata[start + 3][2*i] = i;
      sim->figconstraint.linedata[start + 4][2*i] = i;

      // y
      int nefc = nisland == 1 ? sim->d->nefc : sim->d->island_efcnum[k];
      sim->figconstraint.linedata[start + 0][2*i+1] = nefc;
      const mjSolverStat* stat = sim->d->solver + k*mjNSOLVER + i;
      sim->figconstraint.linedata[start + 1][2*i+1] = stat->nactive;
      sim->figconstraint.linedata[start + 2][2*i+1] = stat->nchange;
      sim->figconstraint.linedata[start + 3][2*i+1] = stat->neval;
      sim->figconstraint.linedata[start + 4][2*i+1] = stat->nupdate;
    }

    // update cost figure
    sim->figcost.linepnt[start + 0] = npoints;
    for (int i=1; i<kCostNum; i++) {
      sim->figcost.linepnt[start + i] = npoints;
    }
    if (sim->m->opt.solver==mjSOL_PGS) {
      sim->figcost.linepnt[start + 1] = 0;
      sim->figcost.linepnt[start + 2] = 0;
    }

    for (int i=0; i<sim->figcost.linepnt[0]; i++) {
      // x
      sim->figcost.linedata[start + 0][2*i] = i;
      sim->figcost.linedata[start + 1][2*i] = i;
      sim->figcost.linedata[start + 2][2*i] = i;

      // y
      const mjSolverStat* stat = sim->d->solver + k*mjNSOLVER + i;
      sim->figcost.linedata[start + 0][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->improvement));
      sim->figcost.linedata[start + 1][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->gradient));
      sim->figcost.linedata[start + 2][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->lineslope));
    }
  }

  // get timers: total, collision, prepare, solve, other
  mjtNum total = sim->d->timer[mjTIMER_STEP].duration;
  int number = sim->d->timer[mjTIMER_STEP].number;
  if (!number) {
    total = sim->d->timer[mjTIMER_FORWARD].duration;
    number = sim->d->timer[mjTIMER_FORWARD].number;
  }
  number = mjMAX(1, number);
  float tdata[5] = {
    static_cast<float>(total/number),
    static_cast<float>(sim->d->timer[mjTIMER_POS_COLLISION].duration/number),
    static_cast<float>(sim->d->timer[mjTIMER_POS_MAKE].duration/number) +
    static_cast<float>(sim->d->timer[mjTIMER_POS_PROJECT].duration/number),
    static_cast<float>(sim->d->timer[mjTIMER_CONSTRAINT].duration/number),
    0
  };
  tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

  // update figtimer
  int pnt = mjMIN(201, sim->figtimer.linepnt[0]+1);
  for (int n=0; n<5; n++) {
    // shift data
    for (int i=pnt-1; i>0; i--) {
      sim->figtimer.linedata[n][2*i+1] = sim->figtimer.linedata[n][2*i-1];
    }

    // assign new
    sim->figtimer.linepnt[n] = pnt;
    sim->figtimer.linedata[n][1] = tdata[n];
  }

  // get total number of iterations and nonzeros
  mjtNum sqrt_nnz = 0;
  int solver_niter = 0;
  for (int island=0; island < nisland; island++) {
    sqrt_nnz += mju_sqrt(sim->d->solver_nnz[island]);
    solver_niter += sim->d->solver_niter[island];
  }

  // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
  float sdata[6] = {
    static_cast<float>(sim->m->nv),
    static_cast<float>(sim->m->nbody),
    static_cast<float>(sim->d->nefc),
    static_cast<float>(sqrt_nnz),
    static_cast<float>(sim->d->ncon),
    static_cast<float>(solver_niter)
  };

  // update figsize
  pnt = mjMIN(201, sim->figsize.linepnt[0]+1);
  for (int n=0; n<6; n++) {
    // shift data
    for (int i=pnt-1; i>0; i--) {
      sim->figsize.linedata[n][2*i+1] = sim->figsize.linedata[n][2*i-1];
    }

    // assign new
    sim->figsize.linepnt[n] = pnt;
    sim->figsize.linedata[n][1] = sdata[n];
  }
}

// show profiler figures
void ShowProfiler(mj::Simulate* sim, mjrRect rect) {
  mjrRect viewport = {
    rect.left + rect.width - rect.width/4,
    rect.bottom,
    rect.width/4,
    rect.height/4
  };
  mjr_figure(viewport, &sim->figtimer, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figsize, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figcost, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figconstraint, &sim->platform_ui->mjr_context());
}


// init sensor figure
void InitializeSensor(mj::Simulate* sim) {
  mjvFigure& figsensor = sim->figsensor;

  // set figure to default
  mjv_defaultFigure(&figsensor);
  figsensor.figurergba[3] = 0.5f;

  // set flags
  figsensor.flg_extend = 1;
  figsensor.flg_barplot = 1;
  figsensor.flg_symmetric = 1;

  // title
  mju::strcpy_arr(figsensor.title, "Sensor data");

  // y-tick nubmer format
  mju::strcpy_arr(figsensor.yformat, "%.0f");

  // grid size
  figsensor.gridsize[0] = 2;
  figsensor.gridsize[1] = 3;

  // minimum range
  figsensor.range[0][0] = 0;
  figsensor.range[0][1] = 0;
  figsensor.range[1][0] = -1;
  figsensor.range[1][1] = 1;
}

// update sensor figure
void UpdateSensor(mj::Simulate* sim) {
  mjModel* m = sim->m;
  mjvFigure& figsensor = sim->figsensor;
  static const int maxline = 10;

  // clear linepnt
  for (int i=0; i<maxline; i++) {
    figsensor.linepnt[i] = 0;
  }

  // start with line 0
  int lineid = 0;

  // loop over sensors
  for (int n=0; n<m->nsensor; n++) {
    // go to next line if type is different
    if (n>0 && m->sensor_type[n]!=m->sensor_type[n-1]) {
      lineid = mjMIN(lineid+1, maxline-1);
    }

    // get info about this sensor
    mjtNum cutoff = (m->sensor_cutoff[n]>0 ? m->sensor_cutoff[n] : 1);
    int adr = m->sensor_adr[n];
    int dim = m->sensor_dim[n];

    // data pointer in line
    int p = figsensor.linepnt[lineid];

    // fill in data for this sensor
    for (int i=0; i<dim; i++) {
      // check size
      if ((p+2*i)>=mjMAXLINEPNT/2) {
        break;
      }

      // x
      figsensor.linedata[lineid][2*p+4*i] = adr+i;
      figsensor.linedata[lineid][2*p+4*i+2] = adr+i;

      // y
      figsensor.linedata[lineid][2*p+4*i+1] = 0;
      figsensor.linedata[lineid][2*p+4*i+3] = sim->d->sensordata[adr+i]/cutoff;
    }

    // update linepnt
    figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                       figsensor.linepnt[lineid]+2*dim);
  }
}

// show sensor figure
void ShowSensor(mj::Simulate* sim, mjrRect rect) {
  // constant width with and without profiler
  int width = sim->profiler ? rect.width/3 : rect.width/4;

  // render figure on the right
  mjrRect viewport = {
    rect.left + rect.width - width,
    rect.bottom,
    width,
    rect.height/3
  };
  mjr_figure(viewport, &sim->figsensor, &sim->platform_ui->mjr_context());
}

// prepare info text
void UpdateInfoText(mj::Simulate* sim,
                    char (&title)[mj::Simulate::kMaxFilenameLength],
                    char (&content)[mj::Simulate::kMaxFilenameLength],
                    double interval) {
  mjModel* m = sim->m;
  mjData* d = sim->d;

  // compute solver error
  int island = 0;  // first island only
  mjtNum solerr = 0;
  if (d->solver_niter[island]) {
    int ind = mjMIN(sim->d->solver_niter[island]-1, mjNSOLVER-1);
    const mjSolverStat* stat = sim->d->solver + island*mjNSOLVER + ind;
    solerr = mju_min(stat->improvement, stat->gradient);
    if (solerr == 0) {
      solerr = mju_max(stat->improvement, stat->gradient);
    }
  }
  solerr = mju_log10(mju_max(mjMINVAL, solerr));

  // prepare info text
  mju::strcpy_arr(title, "Objective\nDoFs\nControls\nParameters\nTime\nMemory");
  const mjpc::Trajectory* best_trajectory =
      sim->agent->ActivePlanner().BestTrajectory();
  if (best_trajectory) {
    int nparam = sim->agent->ActivePlanner().NumParameters();
    mju::sprintf_arr(content, "%.3f\n%d\n%d\n%d\n%-9.3f\n%.2g of %s",
                     best_trajectory->total_return, m->nv, m->nu, nparam,
                     d->time, d->maxuse_arena / (double)(d->narena),
                     mju_writeNumBytes(d->narena));
  }

  // add Energy if enabled
  {
    if (mjENABLED(mjENBL_ENERGY)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%.3f", d->energy[0]+d->energy[1]);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if (mjENABLED(mjENBL_FWDINV)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%.1f %.1f",
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[0])),
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[1])));
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nFwdInv");
    }

    // add islands if enabled
    if (mjENABLED(mjENBL_ISLAND)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%d", d->nisland);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nIslands");
    }
  }
}

// sprintf forwarding, to avoid compiler warning in x-macro
void PrintField(char (&str)[mjMAXUINAME], void* ptr) {
  mju::sprintf_arr(str, "%g", *static_cast<mjtNum*>(ptr));
}

// update watch
void UpdateWatch(mj::Simulate* sim) {
  // clear
  sim->ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
  mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

  // prepare symbols needed by xmacro
  MJDATA_POINTERS_PREAMBLE(sim->m);

  // find specified field in mjData arrays, update value
  #define X(TYPE, NAME, NR, NC)                                                                  \
    if (!mju::strcmp_arr(#NAME, sim->field) &&                                                   \
        !mju::strcmp_arr(#TYPE, "mjtNum")) {                                                     \
      if (sim->index >= 0 && sim->index < sim->m->NR * NC) {                                     \
        PrintField(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], sim->d->NAME + sim->index);  \
      } else {                                                                                   \
        mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid index");       \
      }                                                                                          \
      return;                                                                                    \
    }

  MJDATA_POINTERS
#undef X
}

//---------------------------------- dashboard rendering -------------------------------------------

// 全局仪表盘数据
static mjpc::DashboardData g_dashboard_data;
static mjpc::DashboardDataExtractor* g_dashboard_extractor = nullptr;


// 绘制圆形
void DrawCircle(float cx, float cy, float r, int segments) {
  glBegin(GL_TRIANGLE_FAN);
  glVertex2f(cx, cy);
  for (int i = 0; i <= segments; i++) {
    float angle = 2.0f * M_PI * i / segments;
    glVertex2f(cx + r * cos(angle), cy + r * sin(angle));
  }
  glEnd();
}

// 绘制圆弧
void DrawArc(float cx, float cy, float r, float start_angle, float end_angle, int segments) {
  glBegin(GL_TRIANGLE_FAN);
  glVertex2f(cx, cy);
  for (int i = 0; i <= segments; i++) {
    float t = (float)i / segments;
    float angle = start_angle + t * (end_angle - start_angle);
    glVertex2f(cx + r * cos(angle), cy + r * sin(angle));
  }
  glEnd();
}

// 绘制矩形
void DrawRect(float x, float y, float width, float height) {
  glBegin(GL_QUADS);
  glVertex2f(x, y);
  glVertex2f(x + width, y);
  glVertex2f(x + width, y + height);
  glVertex2f(x, y + height);
  glEnd();
}

// 绘制线
void DrawLine(float x1, float y1, float x2, float y2) {
  glBegin(GL_LINES);
  glVertex2f(x1, y1);
  glVertex2f(x2, y2);
  glEnd();
}

// 渲染仪表盘
void RenderDashboard(const mjpc::DashboardData& data, mjrRect rect, mjrContext* con) {
  // 保存当前OpenGL状态
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  
  // 设置2D正交投影
  glOrtho(0, rect.width, 0, rect.height, -1, 1);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  
  // 禁用深度测试和光照
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  
  // 启用混合（透明效果）
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  // 辅助函数：绘制更好的数字（使用更清晰的7段数码管风格）
  auto DrawNumber = [](float x, float y, const char* str, float r, float g, float b, float scale = 0.8f) {
    glColor4f(r, g, b, 1.0f);
    glLineWidth(1.5f * scale);  // 稍微加粗线条
    
    for (int i = 0; str[i] != '\0'; i++) {
      char ch = str[i];
      float char_x = x + i * 10 * scale;  // 增加字符间距
      float char_width = 8 * scale;
      float char_height = 12 * scale;
      
      // 7段数码管的位置定义
      float seg_length = char_width * 0.8f;
      float seg_thickness = 1.5f * scale;
      float center_x = char_x + char_width/2;
      float top_y = y + char_height;
      float middle_y = y + char_height/2;
      float bottom_y = y;
      
      // 根据字符绘制相应的段
      switch (ch) {
        case '0':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 左上竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, middle_y);
          // 左下竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, middle_y, 
                   center_x - seg_length/2 + seg_thickness/2, bottom_y + seg_thickness);
          // 右上竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, middle_y);
          // 右下竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, middle_y, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '1':
          // 右上竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          // 顶部斜线
          DrawLine(center_x - seg_length/4, top_y - seg_thickness/3, 
                   center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness);
          // 底部横线
          DrawLine(center_x - seg_length/4, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness/2);
          break;
          
        case '2':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 右上竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, middle_y + seg_thickness/2);
          // 左下竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, middle_y - seg_thickness/2, 
                   center_x - seg_length/2 + seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '3':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 右竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '4':
          // 左上竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, middle_y, 
                   center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness);
          // 右竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          break;
          
        case '5':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 左上竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, middle_y - seg_thickness/2);
          // 右下竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, middle_y + seg_thickness/2, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '6':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 左竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, bottom_y + seg_thickness);
          // 右下竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, middle_y + seg_thickness/2, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '7':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 右斜线
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '8':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 左竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, bottom_y + seg_thickness);
          // 右竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
          
        case '9':
          // 上横
          DrawLine(center_x - seg_length/2, top_y - seg_thickness/2, 
                   center_x + seg_length/2, top_y - seg_thickness/2);
          // 中横
          DrawLine(center_x - seg_length/2, middle_y, 
                   center_x + seg_length/2, middle_y);
          // 下横
          DrawLine(center_x - seg_length/2, bottom_y + seg_thickness/2, 
                   center_x + seg_length/2, bottom_y + seg_thickness/2);
          // 左上竖
          DrawLine(center_x - seg_length/2 + seg_thickness/2, top_y - seg_thickness, 
                   center_x - seg_length/2 + seg_thickness/2, middle_y - seg_thickness/2);
          // 右竖
          DrawLine(center_x + seg_length/2 - seg_thickness/2, top_y - seg_thickness, 
                   center_x + seg_length/2 - seg_thickness/2, bottom_y + seg_thickness);
          break;
      }
    }
    glLineWidth(1.0f);  // 恢复线宽
  };

  // 辅助函数：绘制小数字（用于刻度值）
  auto DrawSmallNumber = [&DrawNumber](float x, float y, const char* str, float r, float g, float b) {
    DrawNumber(x, y, str, r, g, b, 0.5f);  // 使用0.5倍缩放
  };
  
  // 1. 绘制速度表
  float speed_cx = 100;
  float speed_cy = rect.height - 150;
  float speed_radius = 50;

  // 速度表参数 - 开口正对下方，240度范围，对称于Y轴
  const float SPEED_MAX_DISPLAY = 10.0f;
  const float SPEED_ANGLE_RANGE = 240.0f * M_PI / 180.0f;
  const float SPEED_START_ANGLE = 210.0f * M_PI / 180.0f; 
  const float SPEED_END_ANGLE = SPEED_START_ANGLE - SPEED_ANGLE_RANGE;

  // 速度表背景（金属质感灰色）
  glColor4f(0.12f, 0.12f, 0.12f, 0.8f);
  DrawCircle(speed_cx, speed_cy, speed_radius, 36);

  // 添加金属边框
  glColor4f(0.4f, 0.4f, 0.4f, 1.0f);
  glLineWidth(2.0f);
  for (int i = 0; i < 2; i++) {
    float r = speed_radius + i - 1;
    DrawCircle(speed_cx, speed_cy, r, 72);
  }
  glLineWidth(1.0f);

  // 主刻度线（每2 km/h一个大刻度） - 顺时针排列
  glColor4f(0.9f, 0.9f, 0.9f, 1.0f);
  glLineWidth(2.0f);
  for (int i = 0; i <= 10; i += 2) {
    float ratio = i / SPEED_MAX_DISPLAY;
    float angle = SPEED_START_ANGLE - ratio * SPEED_ANGLE_RANGE;
    
    // 主刻度线
    float r1 = speed_radius * 0.7f;
    float r2 = speed_radius * 0.95f;
    
    float x1 = speed_cx + r1 * cos(angle);
    float y1 = speed_cy + r1 * sin(angle);
    float x2 = speed_cx + r2 * cos(angle);
    float y2 = speed_cy + r2 * sin(angle);
    
    DrawLine(x1, y1, x2, y2);
    
    // 添加刻度值显示
    char value_text[8];
    snprintf(value_text, sizeof(value_text), "%d", i);
    
    // 计算刻度值位置（稍微远离刻度线，更加向外）
    float text_radius = speed_radius * 0.75f;  // 增加半径，使文字更靠外
    float text_x = speed_cx + text_radius * cos(angle);
    float text_y = speed_cy + text_radius * sin(angle);
    
    // 根据位置智能调整文本对齐，使数字始终朝向圆心
    float offset_x = 0, offset_y = 0;
    float char_width = 8 * 0.5f;  // 小数字的宽度
    float char_height = 12 * 0.5f;  // 小数字的高度
    
    // 计算从圆心到文字位置的方向
    float dir_x = text_x - speed_cx;
    float dir_y = text_y - speed_cy;
    float dir_length = sqrt(dir_x*dir_x + dir_y*dir_y);
    
    if (dir_length > 0) {
      dir_x /= dir_length;
      dir_y /= dir_length;
      
      // 根据方向计算偏移，使数字中心对准刻度线末端
      // 偏移量基于数字大小和方向
      offset_x = -dir_x * (char_width * strlen(value_text) / 2);
      offset_y = -dir_y * (char_height / 2);
      
      // 额外微调，防止数字重叠到表盘内
      offset_x += dir_y * 2;  // 稍微旋转以避免完全径向
      offset_y -= dir_x * 2;
    }
    
    // 绘制数字（使用更好的字体）
    glLineWidth(1.0f);
    DrawSmallNumber(text_x + offset_x, text_y + offset_y, value_text, 0.95f, 0.95f, 0.95f);
    glLineWidth(2.0f);
  }
  glLineWidth(1.0f);

  // 次刻度线（每1 km/h一个小刻度） - 顺时针排列
  glColor4f(0.7f, 0.7f, 0.7f, 0.7f);
  for (int i = 1; i < 10; i += 1) {
    if (i % 2 == 0) continue;  // 跳过主刻度位置
    
    float ratio = i / SPEED_MAX_DISPLAY;
    float angle = SPEED_START_ANGLE - ratio * SPEED_ANGLE_RANGE;
    
    float r1 = speed_radius * 0.8f;
    float r2 = speed_radius * 0.9f;
    
    float x1 = speed_cx + r1 * cos(angle);
    float y1 = speed_cy + r1 * sin(angle);
    float x2 = speed_cx + r2 * cos(angle);
    float y2 = speed_cy + r2 * sin(angle);
    
    DrawLine(x1, y1, x2, y2);
  }

  // 绘制开口标记线（用白色细线表示刻度范围边界）
  glColor4f(0.8f, 0.8f, 0.8f, 0.5f);
  glLineWidth(1.0f);

  // 左侧起始边界线
  float start_boundary_x = speed_cx + speed_radius * 0.9f * cos(SPEED_START_ANGLE);
  float start_boundary_y = speed_cy + speed_radius * 0.9f * sin(SPEED_START_ANGLE);
  DrawLine(speed_cx, speed_cy, start_boundary_x, start_boundary_y);

  // 右侧结束边界线
  float end_boundary_x = speed_cx + speed_radius * 0.9f * cos(SPEED_END_ANGLE);
  float end_boundary_y = speed_cy + speed_radius * 0.9f * sin(SPEED_END_ANGLE);
  DrawLine(speed_cx, speed_cy, end_boundary_x, end_boundary_y);

  // 绘制开口处的弧线（表示开口位置）
  glColor4f(0.6f, 0.6f, 0.6f, 0.3f);
  glLineWidth(1.0f);
  DrawArc(speed_cx, speed_cy, speed_radius * 0.9f, SPEED_END_ANGLE, SPEED_START_ANGLE, 20);

  // 速度颜色区域指示 - 顺时针排列
  float speed_safe_ratio = 6.0f / SPEED_MAX_DISPLAY;
  float speed_safe_end_angle = SPEED_START_ANGLE - speed_safe_ratio * SPEED_ANGLE_RANGE;
  glColor4f(0.0f, 0.6f, 0.0f, 0.2f);
  DrawArc(speed_cx, speed_cy, speed_radius * 0.88f, speed_safe_end_angle, SPEED_START_ANGLE, 40);

  float speed_warning_ratio = 8.0f / SPEED_MAX_DISPLAY;
  float speed_warning_end_angle = SPEED_START_ANGLE - speed_warning_ratio * SPEED_ANGLE_RANGE;
  glColor4f(1.0f, 1.0f, 0.0f, 0.2f);
  DrawArc(speed_cx, speed_cy, speed_radius * 0.88f, speed_warning_end_angle, speed_safe_end_angle, 40);

  float speed_danger_end_angle = SPEED_END_ANGLE;
  glColor4f(1.0f, 0.0f, 0.0f, 0.2f);
  DrawArc(speed_cx, speed_cy, speed_radius * 0.88f, speed_danger_end_angle, speed_warning_end_angle, 40);

  // 速度表指针（顺时针旋转）
  float speed_ratio = data.speed_kmh / SPEED_MAX_DISPLAY;
  if (speed_ratio > 1.0f) speed_ratio = 1.0f;
  float speed_angle = SPEED_START_ANGLE - speed_ratio * SPEED_ANGLE_RANGE;

  // 指针主体（三角形，渐变颜色）
  glBegin(GL_TRIANGLES);
  float speed_tip_x = speed_cx + speed_radius * 0.85f * cos(speed_angle);
  float speed_tip_y = speed_cy + speed_radius * 0.85f * sin(speed_angle);

  if (data.speed_kmh < 6.0f) {
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
  } else if (data.speed_kmh < 8.0f) {
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
  } else {
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
  }
  glVertex2f(speed_tip_x, speed_tip_y);

  float speed_pointer_width = 3.5f;
  float speed_perp_angle = speed_angle + M_PI / 2.0f;
  float speed_base_radius = speed_radius * 0.15f;

  float speed_base_x1 = speed_cx + speed_base_radius * cos(speed_angle) + speed_pointer_width * cos(speed_perp_angle);
  float speed_base_y1 = speed_cy + speed_base_radius * sin(speed_angle) + speed_pointer_width * sin(speed_perp_angle);
  float speed_base_x2 = speed_cx + speed_base_radius * cos(speed_angle) - speed_pointer_width * cos(speed_perp_angle);
  float speed_base_y2 = speed_cy + speed_base_radius * sin(speed_angle) - speed_pointer_width * sin(speed_perp_angle);

  glColor4f(0.6f, 0.6f, 0.6f, 1.0f);
  glVertex2f(speed_base_x1, speed_base_y1);
  glVertex2f(speed_base_x2, speed_base_y2);
  glEnd();

  // 指针中心装饰
  glColor4f(0.8f, 0.8f, 0.8f, 1.0f);
  DrawCircle(speed_cx, speed_cy, 7, 24);
  glColor4f(0.1f, 0.1f, 0.1f, 1.0f);
  DrawCircle(speed_cx, speed_cy, 4, 20);

// 速度表标题 - 移到速度表下方（原来在上方）
glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
glLineWidth(1.5f);
// 修改位置：从上方 speed_cy - speed_radius - 45 改为下方 speed_cy - speed_radius - 25
DrawNumber(speed_cx - 20, speed_cy - speed_radius - 25, "SPEED", 1.0f, 1.0f, 1.0f, 0.6f);
glLineWidth(1.0f);

// 当前速度数值显示（移到速度表上方）
char speed_value_text[16];
snprintf(speed_value_text, sizeof(speed_value_text), "%.1f", data.speed_kmh);

if (data.speed_kmh < 6.0f) {
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
} else if (data.speed_kmh < 8.0f) {
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
} else {
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
}

float speed_text_width = strlen(speed_value_text) * 10 * 0.8f;  // 计算文字宽度
float speed_text_x = speed_cx - speed_text_width / 2;  // 居中

// 修改位置：移到速度表上方，从下方改为上方 +50 像素
DrawNumber(speed_text_x, speed_cy + speed_radius + 30, speed_value_text, 
           data.speed_kmh < 6.0f ? 0.0f : (data.speed_kmh < 8.0f ? 1.0f : 1.0f),
           data.speed_kmh < 6.0f ? 1.0f : (data.speed_kmh < 8.0f ? 1.0f : 0.0f),
           0.0f, 0.8f);

// 单位标签也移到上方
glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
// 修改位置：从下方改为速度值上方 +15 像素
DrawNumber(speed_cx - 15, speed_cy + speed_radius + 45, "km/h", 0.7f, 0.7f, 0.7f, 0.5f);
  
  // 2. 绘制转速表
  float rpm_cx = 250;
  float rpm_cy = rect.height - 150;
  float rpm_radius = 50;

  const float RPM_MAX = 2000.0f;
  const float RPM_ANGLE_RANGE = 240.0f * M_PI / 180.0f;
  const float RPM_START_ANGLE = 210.0f * M_PI / 180.0f;
  const float RPM_END_ANGLE = RPM_START_ANGLE - RPM_ANGLE_RANGE;

  // 转速表背景
  glColor4f(0.12f, 0.12f, 0.14f, 0.8f);
  DrawCircle(rpm_cx, rpm_cy, rpm_radius, 36);

  // 添加金属边框
  glColor4f(0.35f, 0.35f, 0.45f, 1.0f);
  glLineWidth(2.0f);
  for (int i = 0; i < 2; i++) {
    float r = rpm_radius + i - 1;
    DrawCircle(rpm_cx, rpm_cy, r, 72);
  }
  glLineWidth(1.0f);

  // 主刻度线（每500 RPM一个大刻度）
  glColor4f(0.9f, 0.9f, 0.9f, 1.0f);
  glLineWidth(2.0f);
  for (int i = 0; i <= 4; i++) {
    float rpm_value = i * 500.0f;
    float ratio = rpm_value / RPM_MAX;
    float angle = RPM_START_ANGLE - ratio * RPM_ANGLE_RANGE;
    
    float r1 = rpm_radius * 0.7f;
    float r2 = rpm_radius * 0.95f;
    
    float x1 = rpm_cx + r1 * cos(angle);
    float y1 = rpm_cy + r1 * sin(angle);
    float x2 = rpm_cx + r2 * cos(angle);
    float y2 = rpm_cy + r2 * sin(angle);
    
    DrawLine(x1, y1, x2, y2);
    
    // 添加转速刻度值显示
    char rpm_value_text[8];
    snprintf(rpm_value_text, sizeof(rpm_value_text), "%.0f", rpm_value);
    
    // 计算刻度值位置（更加向外）
    float text_radius = rpm_radius * 0.75f;  // 增加半径
    float text_x = rpm_cx + text_radius * cos(angle);
    float text_y = rpm_cy + text_radius * sin(angle);
    
    // 根据位置智能调整文本对齐
    float offset_x = 0, offset_y = 0;
    float char_width = 8 * 0.5f;
    float char_height = 12 * 0.5f;
    
    // 计算从圆心到文字位置的方向
    float dir_x = text_x - rpm_cx;
    float dir_y = text_y - rpm_cy;
    float dir_length = sqrt(dir_x*dir_x + dir_y*dir_y);
    
    if (dir_length > 0) {
      dir_x /= dir_length;
      dir_y /= dir_length;
      
      // 根据方向计算偏移
      offset_x = -dir_x * (char_width * strlen(rpm_value_text) / 2);
      offset_y = -dir_y * (char_height / 2);
      
      // 额外微调
      offset_x += dir_y * 2;
      offset_y -= dir_x * 2;
    }
    
    // 绘制数字
    glLineWidth(1.0f);
    DrawSmallNumber(text_x + offset_x, text_y + offset_y, rpm_value_text, 0.95f, 0.95f, 0.95f);
    glLineWidth(2.0f);
  }
  glLineWidth(1.0f);

  // 次刻度线（每250 RPM一个小刻度）
  glColor4f(0.7f, 0.7f, 0.7f, 0.7f);
  for (int i = 1; i < 8; i += 1) {
    float rpm_value = i * 250.0f;
    if (rpm_value > RPM_MAX) break;
    
    float ratio = rpm_value / RPM_MAX;
    float angle = RPM_START_ANGLE - ratio * RPM_ANGLE_RANGE;
    
    float r1 = rpm_radius * 0.8f;
    float r2 = rpm_radius * 0.9f;
    
    float x1 = rpm_cx + r1 * cos(angle);
    float y1 = rpm_cy + r1 * sin(angle);
    float x2 = rpm_cx + r2 * cos(angle);
    float y2 = rpm_cy + r2 * sin(angle);
    
    DrawLine(x1, y1, x2, y2);
  }

  // 绘制转速表开口标记线
  glColor4f(0.5f, 0.5f, 0.8f, 0.5f);
  glLineWidth(1.0f);
  float rpm_start_boundary_x = rpm_cx + rpm_radius * 0.9f * cos(RPM_START_ANGLE);
  float rpm_start_boundary_y = rpm_cy + rpm_radius * 0.9f * sin(RPM_START_ANGLE);
  DrawLine(rpm_cx, rpm_cy, rpm_start_boundary_x, rpm_start_boundary_y);
  float rpm_end_boundary_x = rpm_cx + rpm_radius * 0.9f * cos(RPM_END_ANGLE);
  float rpm_end_boundary_y = rpm_cy + rpm_radius * 0.9f * sin(RPM_END_ANGLE);
  DrawLine(rpm_cx, rpm_cy, rpm_end_boundary_x, rpm_end_boundary_y);

  // 绘制开口处的弧线
  glColor4f(0.4f, 0.4f, 0.6f, 0.3f);
  glLineWidth(1.0f);
  DrawArc(rpm_cx, rpm_cy, rpm_radius * 0.9f, RPM_END_ANGLE, RPM_START_ANGLE, 20);

  // 转速颜色区域指示
  float rpm_safe_ratio = 1500.0f / RPM_MAX;
  float rpm_safe_end_angle = RPM_START_ANGLE - rpm_safe_ratio * RPM_ANGLE_RANGE;
  glColor4f(0.0f, 0.6f, 0.0f, 0.2f);
  DrawArc(rpm_cx, rpm_cy, rpm_radius * 0.88f, rpm_safe_end_angle, RPM_START_ANGLE, 40);

  float rpm_warning_ratio = 1800.0f / RPM_MAX;
  float rpm_warning_end_angle = RPM_START_ANGLE - rpm_warning_ratio * RPM_ANGLE_RANGE;
  glColor4f(1.0f, 1.0f, 0.0f, 0.2f);
  DrawArc(rpm_cx, rpm_cy, rpm_radius * 0.88f, rpm_warning_end_angle, rpm_safe_end_angle, 40);

  float rpm_danger_end_angle = RPM_END_ANGLE;
  glColor4f(1.0f, 0.0f, 0.0f, 0.2f);
  DrawArc(rpm_cx, rpm_cy, rpm_radius * 0.88f, rpm_danger_end_angle, rpm_warning_end_angle, 40);

  // 转速表指针
  float rpm_ratio = data.rpm / RPM_MAX;
  if (rpm_ratio > 1.0f) rpm_ratio = 1.0f;
  float rpm_angle = RPM_START_ANGLE - rpm_ratio * RPM_ANGLE_RANGE;

  // 指针主体
  glBegin(GL_TRIANGLES);
  float rpm_tip_x = rpm_cx + rpm_radius * 0.85f * cos(rpm_angle);
  float rpm_tip_y = rpm_cy + rpm_radius * 0.85f * sin(rpm_angle);

  if (data.rpm < 1500.0f) {
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
  } else if (data.rpm < 1800.0f) {
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
  } else {
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
  }
  glVertex2f(rpm_tip_x, rpm_tip_y);

  float rpm_pointer_width = 3.5f;
  float rpm_perp_angle = rpm_angle + M_PI / 2.0f;
  float rpm_base_radius = rpm_radius * 0.15f;

  float rpm_base_x1 = rpm_cx + rpm_base_radius * cos(rpm_angle) + rpm_pointer_width * cos(rpm_perp_angle);
  float rpm_base_y1 = rpm_cy + rpm_base_radius * sin(rpm_angle) + rpm_pointer_width * sin(rpm_perp_angle);
  float rpm_base_x2 = rpm_cx + rpm_base_radius * cos(rpm_angle) - rpm_pointer_width * cos(rpm_perp_angle);
  float rpm_base_y2 = rpm_cy + rpm_base_radius * sin(rpm_angle) - rpm_pointer_width * sin(rpm_perp_angle);

  glColor4f(0.6f, 0.6f, 0.6f, 1.0f);
  glVertex2f(rpm_base_x1, rpm_base_y1);
  glVertex2f(rpm_base_x2, rpm_base_y2);
  glEnd();

  // 指针中心装饰
  glColor4f(0.8f, 0.8f, 0.8f, 1.0f);
  DrawCircle(rpm_cx, rpm_cy, 7, 24);
  glColor4f(0.1f, 0.1f, 0.1f, 1.0f);
  DrawCircle(rpm_cx, rpm_cy, 4, 20);

// 转速表标题 - 移到转速表下方（原来在上方）
glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
glLineWidth(1.5f);
// 修改位置：从上方 rpm_cy - rpm_radius - 45 改为下方 rpm_cy - rpm_radius - 25
DrawNumber(rpm_cx - 15, rpm_cy - rpm_radius - 25, "RPM", 1.0f, 1.0f, 1.0f, 0.6f);
glLineWidth(1.0f);

// 当前转速数值显示（移到转速表上方）
char current_rpm_text[16];
snprintf(current_rpm_text, sizeof(current_rpm_text), "%.0f", data.rpm);

float rpm_text_width = strlen(current_rpm_text) * 10 * 0.8f;
float rpm_text_x = rpm_cx - rpm_text_width / 2;

if (data.rpm < 1500.0f) {
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
} else if (data.rpm < 1800.0f) {
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
} else {
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
}

// 修改位置：移到转速表上方，从下方改为上方 +30 像素
DrawNumber(rpm_text_x, rpm_cy + rpm_radius + 30, current_rpm_text, 
           data.rpm < 1500.0f ? 0.0f : (data.rpm < 1800.0f ? 1.0f : 1.0f),
           data.rpm < 1500.0f ? 1.0f : (data.rpm < 1800.0f ? 1.0f : 0.0f),
           0.0f, 0.8f);
  
  // 3. 绘制油量显示
  float fuel_x = 50;
  float fuel_y = rect.height - 250; 
  float fuel_width = 100;
  float fuel_height = 20;
  
  // 油量背景
  glColor4f(0.1f, 0.1f, 0.1f, 0.3f);
  DrawRect(fuel_x, fuel_y, fuel_width, fuel_height);
  
  // 油量条
  float fuel_ratio = data.fuel / 100.0f;
  if (fuel_ratio < 0) fuel_ratio = 0;
  if (fuel_ratio > 1) fuel_ratio = 1;
  
  if (fuel_ratio > 0.3) {
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
  } else {
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
  }
  DrawRect(fuel_x + 2, fuel_y + 2, (fuel_width - 4) * fuel_ratio, fuel_height - 4);
  
  // 油量标签
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glLineWidth(1.5f);
  DrawNumber(fuel_x, fuel_y - 20, "FUEL", 1.0f, 1.0f, 1.0f, 0.5f);
  glLineWidth(1.0f);
  
  // 4. 绘制温度显示
  float temp_x = 200;
  float temp_y = rect.height - 250;
  float temp_width = 100;
  float temp_height = 20;
  
  // 温度背景
  glColor4f(0.1f, 0.1f, 0.1f, 0.3f);
  DrawRect(temp_x, temp_y, temp_width, temp_height);
  
  // 温度条
  float temp_ratio = (data.temperature - 60.0f) / 60.0f;
  if (temp_ratio < 0) temp_ratio = 0;
  if (temp_ratio > 1) temp_ratio = 1;
  
  if (temp_ratio < 0.5) {
    float t = temp_ratio * 2.0f;
    glColor4f(0.0f, t, 1.0f - t, 1.0f);
  } else {
    float t = (temp_ratio - 0.5f) * 2.0f;
    glColor4f(t, 1.0f - t, 0.0f, 1.0f);
  }
  DrawRect(temp_x + 2, temp_y + 2, (temp_width - 4) * temp_ratio, temp_height - 4);
  
  // 温度标签
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glLineWidth(1.5f);
  DrawNumber(temp_x, temp_y - 20, "TEMP", 1.0f, 1.0f, 1.0f, 0.5f);
  glLineWidth(1.0f);
  
  // 恢复状态
  glDisable(GL_BLEND);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopAttrib();
}
  

//---------------------------------- UI construction -----------------------------------------------

// make physics section of UI
void MakePhysicsSection(mj::Simulate* sim, int oldstate) {
  mjOption* opt = &sim->m->opt;

  mjuiDef defPhysics[] = {
    {mjITEM_SECTION,   "Physics",       oldstate, nullptr,           "AP"},
    {mjITEM_SELECT,    "Integrator",    2, &(opt->integrator),        "Euler\nRK4\nimplicit\nimplicitfast"},
    {mjITEM_SELECT,    "Cone",          2, &(opt->cone),              "Pyramidal\nElliptic"},
    {mjITEM_SELECT,    "Jacobian",      2, &(opt->jacobian),          "Dense\nSparse\nAuto"},
    {mjITEM_SELECT,    "Solver",        2, &(opt->solver),            "PGS\nCG\nNewton"},
    {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
    {mjITEM_EDITNUM,   "Timestep",      2, &(opt->timestep),          "1 0 1"},
    {mjITEM_EDITINT,   "Iterations",    2, &(opt->iterations),        "1 0 1000"},
    {mjITEM_EDITNUM,   "Tolerance",     2, &(opt->tolerance),         "1 0 1"},
    {mjITEM_EDITINT,   "LS Iter",       2, &(opt->ls_iterations),     "1 0 100"},
    {mjITEM_EDITNUM,   "LS Tol",        2, &(opt->ls_tolerance),      "1 0 0.1"},
    {mjITEM_EDITINT,   "Noslip Iter",   2, &(opt->noslip_iterations), "1 0 1000"},
    {mjITEM_EDITNUM,   "Noslip Tol",    2, &(opt->noslip_tolerance),  "1 0 1"},
    {mjITEM_EDITINT,   "CCD Iter",      2, &(opt->ccd_iterations),    "1 0 1000"},
    {mjITEM_EDITNUM,   "CCD Tol",       2, &(opt->ccd_tolerance),     "1 0 1"},
    {mjITEM_EDITNUM,   "API Rate",      2, &(opt->apirate),           "1 0 1000"},
    {mjITEM_EDITINT,   "SDF Iter",      2, &(opt->sdf_iterations),    "1 1 20"},
    {mjITEM_EDITINT,   "SDF Init",      2, &(opt->sdf_initpoints),    "1 1 100"},
    {mjITEM_SEPARATOR, "Physical Parameters", 1},
    {mjITEM_EDITNUM,   "Gravity",       2, opt->gravity,              "3"},
    {mjITEM_EDITNUM,   "Wind",          2, opt->wind,                 "3"},
    {mjITEM_EDITNUM,   "Magnetic",      2, opt->magnetic,             "3"},
    {mjITEM_EDITNUM,   "Density",       2, &(opt->density),           "1"},
    {mjITEM_EDITNUM,   "Viscosity",     2, &(opt->viscosity),         "1"},
    {mjITEM_EDITNUM,   "Imp Ratio",     2, &(opt->impratio),          "1"},
    {mjITEM_SEPARATOR, "Disable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defEnableFlags[] = {
    {mjITEM_SEPARATOR, "Enable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defOverride[] = {
    {mjITEM_SEPARATOR, "Contact Override", 1},
    {mjITEM_EDITNUM,   "Margin",        2, &(opt->o_margin),          "1"},
    {mjITEM_EDITNUM,   "Sol Imp",       2, &(opt->o_solimp),          "5"},
    {mjITEM_EDITNUM,   "Sol Ref",       2, &(opt->o_solref),          "2"},
    {mjITEM_END}
  };

  // add physics
  mjui_add(&sim->ui0, defPhysics);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKINT,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (int i=0; i<mjNDISABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjDISABLESTRING[i]);
    defFlag[0].pdata = sim->disable + i;
    mjui_add(&sim->ui0, defFlag);
  }
  mjui_add(&sim->ui0, defEnableFlags);
  for (int i=0; i<mjNENABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjENABLESTRING[i]);
    defFlag[0].pdata = sim->enable + i;
    mjui_add(&sim->ui0, defFlag);
  }

  // add contact override
  mjui_add(&sim->ui0, defOverride);
}



// make rendering section of UI
void MakeRenderingSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defRendering[] = {
      {mjITEM_SECTION, "Rendering", oldstate, nullptr, "AR"},
      {mjITEM_SELECT, "Camera", 2, &(sim->camera), "Free\nTracking"},
      {mjITEM_SELECT, "Label", 2, &(sim->opt.label),
       "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\n"
       "Actuator\nConstraint\nSkin\nSelection\nSel "
       "Pnt\nContact\nForce\nIsland"},
      {mjITEM_SELECT, "Frame", 2, &(sim->opt.frame),
       "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld"},
      {mjITEM_BUTTON, "Copy camera", 2, nullptr, ""},
      {mjITEM_BUTTON, "Copy state", 2, nullptr, ""},
      {mjITEM_SEPARATOR, "Model Elements", 1},
      {mjITEM_END}};
  mjuiDef defOpenGL[] = {
    {mjITEM_SEPARATOR, "OpenGL Effects", 1},
    {mjITEM_END}
  };

  // add model cameras, up to UI limit
  for (int i=0; i<mjMIN(sim->m->ncam, mjMAXUIMULTI-2); i++) {
    // prepare name
    char camname[mjMAXUINAME] = "\n";
    if (sim->m->names[sim->m->name_camadr[i]]) {
      mju::strcat_arr(camname, sim->m->names+sim->m->name_camadr[i]);
    } else {
      mju::sprintf_arr(camname, "\nCamera %d", i);
    }

    // check string length
    if (mju::strlen_arr(camname) + mju::strlen_arr(defRendering[1].other)>=mjMAXUITEXT-1) {
      break;
    }

    // add camera
    mju::strcat_arr(defRendering[1].other, camname);
  }

  // add rendering standard
  mjui_add(&sim->ui0, defRendering);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKBYTE,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (int i=0; i<mjNVISFLAG; i++) {
    // set name, remove "&"
    mju::strcpy_arr(defFlag[0].name, mjVISSTRING[i][0]);
    for (int j=0; j<strlen(mjVISSTRING[i][0]); j++) {
      if (mjVISSTRING[i][0][j]=='&') {
        mju_strncpy(
          defFlag[0].name+j, mjVISSTRING[i][0]+j+1, mju::sizeof_arr(defFlag[0].name)-j);
        break;
      }
    }

    // set shortcut and data
    if (mjVISSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjVISSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->opt.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }

  // create tree slider
  mjuiDef defTree[] = {
      {mjITEM_SLIDERINT, "Tree depth", 2, &sim->opt.bvh_depth, "0 20"},
      {mjITEM_END}
  };
  mjui_add(&sim->ui0, defTree);

  // add rendering flags
  mjui_add(&sim->ui0, defOpenGL);
  for (int i=0; i<mjNRNDFLAG; i++) {
    mju::strcpy_arr(defFlag[0].name, mjRNDSTRING[i][0]);
    if (mjRNDSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->scn.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }
}



// make group section of UI
void MakeGroupSection(mj::Simulate* sim, int oldstate) {
  mjvOption& vopt = sim->opt;
  mjuiDef defGroup[] = {
    {mjITEM_SECTION,    "Group enable",     oldstate, nullptr,          "AG"},
    {mjITEM_SEPARATOR,  "Geom groups",  1},
    {mjITEM_CHECKBYTE,  "Geom 0",           2, vopt.geomgroup,          " 0"},
    {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup+1,        " 1"},
    {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup+2,        " 2"},
    {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup+3,        " 3"},
    {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup+4,        " 4"},
    {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup+5,        " 5"},
    {mjITEM_SEPARATOR,  "Site groups",  1},
    {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
    {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup+1,        "S1"},
    {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup+2,        "S2"},
    {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup+3,        "S3"},
    {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup+4,        "S4"},
    {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup+5,        "S5"},
    {mjITEM_SEPARATOR,  "Joint groups", 1},
    {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
    {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup+1,       ""},
    {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup+2,       ""},
    {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup+3,       ""},
    {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup+4,       ""},
    {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup+5,       ""},
    {mjITEM_SEPARATOR,  "Tendon groups",    1},
    {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
    {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup+1,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup+2,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup+3,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup+4,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup+5,      ""},
    {mjITEM_SEPARATOR,  "Actuator groups", 1},
    {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
    {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup+1,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup+2,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup+3,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup+4,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup+5,    ""},
    {mjITEM_SEPARATOR,  "Skin groups", 1},
    {mjITEM_CHECKBYTE,  "Skin 0",           2, vopt.skingroup,          ""},
    {mjITEM_CHECKBYTE,  "Skin 1",           2, vopt.skingroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Skin 2",           2, vopt.skingroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Skin 3",           2, vopt.skingroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Skin 4",           2, vopt.skingroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Skin 5",           2, vopt.skingroup+5,        ""},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui0, defGroup);
}

// make joint section of UI
void MakeJointSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defJoint[] = {
    {mjITEM_SECTION, "Joint", oldstate, nullptr, "AJ"},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defJoint);
  defSlider[0].state = 4;

  // add scalar joints, exit if UI limit reached
  int itemcnt = 0;
  for (int i=0; i<sim->m->njnt && itemcnt<mjMAXUIITEM; i++)
    if ((sim->m->jnt_type[i]==mjJNT_HINGE || sim->m->jnt_type[i]==mjJNT_SLIDE)) {
      // skip if joint group is disabled
      if (!sim->opt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, sim->m->jnt_group[i]))]) {
        continue;
      }

      // set data and name
      defSlider[0].pdata = sim->d->qpos + sim->m->jnt_qposadr[i];
      if (sim->m->names[sim->m->name_jntadr[i]]) {
        mju::strcpy_arr(defSlider[0].name, sim->m->names+sim->m->name_jntadr[i]);
      } else {
        mju::sprintf_arr(defSlider[0].name, "joint %d", i);
      }

      // set range
      if (sim->m->jnt_limited[i])
        mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                         sim->m->jnt_range[2*i], sim->m->jnt_range[2*i+1]);
      else if (sim->m->jnt_type[i]==mjJNT_SLIDE) {
        mju::strcpy_arr(defSlider[0].other, "-1 1");
      } else {
        mju::strcpy_arr(defSlider[0].other, "-3.1416 3.1416");
      }

      // add and count
      mjui_add(&sim->ui1, defSlider);
      itemcnt++;
    }
}

// make control section of UI
void MakeControlSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defControl[] = {
    {mjITEM_SECTION, "Control", oldstate, nullptr, "AC"},
    {mjITEM_BUTTON,  "Clear all", 2},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defControl);
  defSlider[0].state = 2;

  // add controls, exit if UI limit reached (Clear button already added)
  int itemcnt = 1;
  for (int i=0; i<sim->m->nu && itemcnt<mjMAXUIITEM; i++) {
    // skip if actuator group is disabled
    if (!sim->opt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, sim->m->actuator_group[i]))]) {
      continue;
    }

    // set data and name
    defSlider[0].pdata = sim->d->ctrl + i;
    if (sim->m->names[sim->m->name_actuatoradr[i]]) {
      mju::strcpy_arr(defSlider[0].name, sim->m->names+sim->m->name_actuatoradr[i]);
    } else {
      mju::sprintf_arr(defSlider[0].name, "control %d", i);
    }

    // set range
    if (sim->m->actuator_ctrllimited[i])
      mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                       sim->m->actuator_ctrlrange[2*i], sim->m->actuator_ctrlrange[2*i+1]);
    else {
      mju::strcpy_arr(defSlider[0].other, "-1 1");
    }

    // add and count
    mjui_add(&sim->ui1, defSlider);
    itemcnt++;
  }
}

// make model-dependent UI sections
void MakeUiSections(mj::Simulate* sim) {
  // get section open-close state, UI 0
  int oldstate0[NSECT0];
  for (int i=0; i<NSECT0; i++) {
    oldstate0[i] = 0;
    if (sim->ui0.nsect>i) {
      oldstate0[i] = sim->ui0.sect[i].state;
    }
  }

  // get section open-close state, UI 1
  int oldstate1[NSECT1];
  for (int i=0; i<NSECT1; i++) {
    oldstate1[i] = 0;
    if (sim->ui1.nsect>i) {
      oldstate1[i] = sim->ui1.sect[i].state;
    }
  }

  // clear model-dependent sections of UI
  sim->ui0.nsect = SECT_TASK;
  sim->ui1.nsect = 0;

  // make
  sim->agent->GUI(sim->ui0);
  MakePhysicsSection(sim, oldstate0[SECT_PHYSICS]);
  MakeRenderingSection(sim, oldstate0[SECT_RENDERING]);
  MakeGroupSection(sim, oldstate0[SECT_GROUP]);
  MakeJointSection(sim, oldstate1[SECT_JOINT]);
  MakeControlSection(sim, oldstate1[SECT_CONTROL]);
}

//---------------------------------- utility functions ---------------------------------------------

// align and scale view
void AlignAndScaleView(mj::Simulate* sim) {
  // use default free camera parameters
  mjv_defaultFreeCamera(sim->m, &sim->cam);
}


// copy qpos to clipboard as key
void CopyKey(mj::Simulate* sim) {
  char clipboard[5000] = "<key qpos='";
  char buf[200];

  // prepare string
  for (int i=0; i<sim->m->nq; i++) {
    mju::sprintf_arr(buf, i==sim->m->nq-1 ? "%g" : "%g ", sim->d->qpos[i]);
    mju::strcat_arr(clipboard, buf);
  }
  mju::strcat_arr(clipboard, "'/>");

  // copy to clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// millisecond timer, for MuJoCo built-in profiler
mjtNum Timer() {
  return Milliseconds(mj::Simulate::Clock::now().time_since_epoch()).count();
}

// clear all times
void ClearTimeres(mjData* d) {
  for (int i = 0; i < mjNTIMER; i++) {
    d->timer[i].duration = 0;
    d->timer[i].number = 0;
  }
}

// copy current camera to clipboard as MJCF specification
void CopyCamera(mj::Simulate* sim) {
  mjvGLCamera* camera = sim->scn.camera;

  char clipboard[500];
  mjtNum cam_right[3];
  mjtNum cam_forward[3];
  mjtNum cam_up[3];

  // get camera spec from the GLCamera
  mju_f2n(cam_forward, camera[0].forward, 3);
  mju_f2n(cam_up, camera[0].up, 3);
  mju_cross(cam_right, cam_forward, cam_up);

  // make MJCF camera spec
  mju::sprintf_arr(clipboard,
                   "<camera pos=\"%.3f %.3f %.3f\" xyaxes=\"%.3f %.3f %.3f %.3f %.3f %.3f\"/>\n",
                   (camera[0].pos[0] + camera[1].pos[0]) / 2,
                   (camera[0].pos[1] + camera[1].pos[1]) / 2,
                   (camera[0].pos[2] + camera[1].pos[2]) / 2,
                   cam_right[0], cam_right[1], cam_right[2],
                   camera[0].up[0], camera[0].up[1], camera[0].up[2]);

  // copy spec into clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// update UI 0 when MuJoCo structures change (except for joint sliders)
void UpdateSettings(mj::Simulate* sim) {
  // physics flags
  for (int i=0; i<mjNDISABLE; i++) {
    sim->disable[i] = ((sim->m->opt.disableflags & (1<<i)) !=0);
  }
  for (int i=0; i<mjNENABLE; i++) {
    sim->enable[i] = ((sim->m->opt.enableflags & (1<<i)) !=0);
  }

  // camera
  if (sim->cam.type==mjCAMERA_FIXED) {
    sim->camera = 2 + sim->cam.fixedcamid;
  } else if (sim->cam.type==mjCAMERA_TRACKING) {
    sim->camera = 1;
  } else {
    sim->camera = 0;
  }

  // update UI
  mjui_update(-1, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
}


// Compute suitable font scale.
int ComputeFontScale(const mj::PlatformUIAdapter& platform_ui) {
  // compute framebuffer-to-window ratio
  auto [buf_width, buf_height] = platform_ui.GetFramebufferSize();
  auto [win_width, win_height] = platform_ui.GetWindowSize();
  double b2w = static_cast<double>(buf_width) / win_width;

  // compute PPI
  double PPI = b2w * platform_ui.GetDisplayPixelsPerInch();

  // estimate font scaling, guard against unrealistic PPI
  int fs;
  if (buf_width > win_width) {
    fs = mju_round(b2w * 100);
  } else if (PPI>50 && PPI<350) {
    fs = mju_round(PPI);
  } else {
    fs = 150;
  }
  fs = mju_round(fs * 0.02) * 50;
  fs = mjMIN(300, mjMAX(100, fs));

  return fs;
}

//---------------------------------- UI handlers ---------------------------------------------------

// determine enable/disable item state given category
int UiPredicate(int category, void* userdata) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(userdata);

  switch (category) {
  case 2:                 // require model
    return (sim->m != nullptr);

  case 3:                 // require model and nkey
    return (sim->m && sim->m->nkey);

  case 4:                 // require model and paused
    return (sim->m && !sim->run);

  default:
    return 1;
  }
}

// set window layout
void UiLayout(mjuiState* state) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(state->userdata);
  mjrRect* rect = state->rect;

  // set number of rectangles
  state->nrect = 4;

  // rect 1: UI 0
  rect[1].left = 0;
  rect[1].width = sim->ui0_enable ? sim->ui0.width : 0;
  rect[1].bottom = 0;
  rect[1].height = rect[0].height;

  // rect 2: UI 1
  rect[2].width = sim->ui1_enable ? sim->ui1.width : 0;
  rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
  rect[2].bottom = 0;
  rect[2].height = rect[0].height;

  // rect 3: 3D plot (everything else is an overlay)
  rect[3].left = rect[1].width;
  rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
  rect[3].bottom = 0;
  rect[3].height = rect[0].height;
}

void UiModify(mjUI* ui, mjuiState* state, mjrContext* con) {
  mjui_resize(ui, con);
  mjr_addAux(ui->auxid, ui->width, ui->maxheight, ui->spacing.samples, con);
  UiLayout(state);
  mjui_update(-1, -1, ui, state, con);
}

// handle UI event
void UiEvent(mjuiState* state) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(state->userdata);
  mjModel* m = sim->m;
  mjData* d = sim->d;
  int i;
  char err[200];

  // call UI 0 if event is directed to it
  if ((state->dragrect==sim->ui0.rectid) ||
      (state->dragrect==0 && state->mouserect==sim->ui0.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui0, state, &sim->platform_ui->mjr_context());

    // file section
    if (it && it->sectionid==SECT_FILE) {
      switch (it->itemid) {
      case 0:             // Save xml
        {
          const std::string path = GetSavePath("mjmodel.xml");
          if (!path.empty() && !mj_saveLastXML(path.c_str(), m, err, 200)) {
            std::printf("Save XML error: %s", err);
          }
        }
        break;

      case 1:             // Save mjb
        {
          const std::string path = GetSavePath("mjmodel.mjb");
          if (!path.empty()) {
            mj_saveModel(m, path.c_str(), nullptr, 0);
          }
        }
        break;

      case 2:             // Print model
        mj_printModel(m, "MJMODEL.TXT");
        break;

      case 3:             // Print data
        mj_printData(m, d, "MJDATA.TXT");
        break;

      case 4:             // Quit
        sim->exitrequest.store(1);
        break;

      case 5:             // Screenshot
        sim->screenshotrequest.store(true);
        break;
      }
    } else if (it && it->sectionid == SECT_OPTION) {
      if (it->pdata == &sim->spacing) {
        sim->ui0.spacing = mjui_themeSpacing(sim->spacing);
        sim->ui1.spacing = mjui_themeSpacing(sim->spacing);
      } else if (it->pdata == &sim->color) {
        sim->ui0.color = mjui_themeColor(sim->color);
        sim->ui1.color = mjui_themeColor(sim->color);
      } else if (it->pdata == &sim->font) {
        mjr_changeFont(50 * (sim->font + 1), &sim->platform_ui->mjr_context());
      } else if (it->pdata == &sim->fullscreen) {
        sim->platform_ui->ToggleFullscreen();
      } else if (it->pdata == &sim->vsync) {
        sim->platform_ui->SetVSync(sim->vsync);
      }

      // modify UI
      UiModify(&sim->ui0, state, &sim->platform_ui->mjr_context());
      UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());

    } else if (it && it->sectionid == SECT_SIMULATION) {
      switch (it->itemid) {
      case 1:             // Reset
        if (m) {
          mj_resetDataKeyframe(m, d, mj_name2id(m, mjOBJ_KEY, "home"));
          mj_forward(m, d);
          UpdateProfiler(sim);
          UpdateSensor(sim);
          UpdateSettings(sim);
          sim->agent->PlotReset();
        }
        break;

      case 2:             // Reload
        sim->uiloadrequest.fetch_add(1);
        break;

      case 3:             // Align
        AlignAndScaleView(sim);
        UpdateSettings(sim);
        break;

      case 4:             // Copy pose
        CopyKey(sim);
        break;

      case 5:             // Adjust key
      case 6:             // Load key
        i = sim->key;
        d->time = m->key_time[i];
        mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
        mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
        mju_copy(d->act, m->key_act+i*m->na, m->na);
        mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
        mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
        mju_copy(d->ctrl, m->key_ctrl+i*m->nu, m->nu);
        mj_forward(m, d);
        UpdateProfiler(sim);
        UpdateSensor(sim);
        UpdateSettings(sim);
        break;

      case 7:             // Save key
        i = sim->key;
        m->key_time[i] = d->time;
        mju_copy(m->key_qpos+i*m->nq, d->qpos, m->nq);
        mju_copy(m->key_qvel+i*m->nv, d->qvel, m->nv);
        mju_copy(m->key_act+i*m->na, d->act, m->na);
        mju_copy(m->key_mpos+i*3*m->nmocap, d->mocap_pos, 3*m->nmocap);
        mju_copy(m->key_mquat+i*4*m->nmocap, d->mocap_quat, 4*m->nmocap);
        mju_copy(m->key_ctrl+i*m->nu, d->ctrl, m->nu);
        break;
      }
    }

    // task section
    else if (it && it->sectionid == SECT_TASK) {
      sim->agent->TaskEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // agent section
    else if (it && it->sectionid == SECT_AGENT) {
      sim->agent->AgentEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // estimator section
    else if (it && it->sectionid == SECT_ESTIMATOR) {
      sim->agent->EstimatorEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // physics section
    else if (it && it->sectionid==SECT_PHYSICS) {
      // update disable flags in mjOption
      m->opt.disableflags = 0;
      for (i=0; i<mjNDISABLE; i++)
        if (sim->disable[i]) {
          m->opt.disableflags |= (1<<i);
        }

      // update enable flags in mjOption
      m->opt.enableflags = 0;
      for (i=0; i<mjNENABLE; i++)
        if (sim->enable[i]) {
          m->opt.enableflags |= (1<<i);
        }
    }

    // rendering section
    else if (it && it->sectionid==SECT_RENDERING) {
      // set camera in mjvCamera
      if (sim->camera==0) {
        sim->cam.type = mjCAMERA_FREE;
      } else if (sim->camera==1) {
        if (sim->pert.select>0) {
          sim->cam.type = mjCAMERA_TRACKING;
          sim->cam.trackbodyid = sim->pert.select;
          sim->cam.fixedcamid = -1;
        } else {
          sim->cam.type = mjCAMERA_FREE;
          sim->camera = 0;
          mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate,
                      &sim->platform_ui->mjr_context());
        }
      } else {
        sim->cam.type = mjCAMERA_FIXED;
        sim->cam.fixedcamid = sim->camera - 2;
      }
      // copy camera spec to clipboard (as MJCF element)
      if (it->itemid == 3) {
        CopyCamera(sim);
      }
    }

    // group section
    else if (it && it->sectionid==SECT_GROUP) {
      // remake joint section if joint group changed
      if (it->name[0]=='J' && it->name[1]=='o') {
        sim->ui1.nsect = SECT_JOINT;
        MakeJointSection(sim, sim->ui1.sect[SECT_JOINT].state);
        sim->ui1.nsect = NSECT1;
        UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
      }

      // remake control section if actuator group changed
      if (it->name[0]=='A' && it->name[1]=='c') {
        sim->ui1.nsect = SECT_CONTROL;
        MakeControlSection(sim, sim->ui1.sect[SECT_CONTROL].state);
        sim->ui1.nsect = NSECT1;
        UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // call UI 1 if event is directed to it
  if ((state->dragrect==sim->ui1.rectid) ||
      (state->dragrect==0 && state->mouserect==sim->ui1.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui1, state, &sim->platform_ui->mjr_context());

    // control section
    if (it && it->sectionid==SECT_CONTROL) {
      // clear controls
      if (it->itemid==0) {
        mju_zero(d->ctrl, m->nu);
        mjui_update(SECT_CONTROL, -1, &sim->ui1, &sim->uistate, &sim->platform_ui->mjr_context());
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // shortcut not handled by UI
  if (state->type==mjEVENT_KEY && state->key!=0) {
    switch (state->key) {
    case ' ':                   // Mode
      if (m) {
        sim->run = 1 - sim->run;
        sim->pert.active = 0;
        mjui_update(-1, -1, &sim->ui0, state, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_RIGHT:           // step forward
      if (m && !sim->run) {
        ClearTimeres(d);
        mj_step(m, d);
        UpdateProfiler(sim);
        UpdateSensor(sim);
        UpdateSettings(sim);
      }
      break;

    case mjKEY_PAGE_UP:         // select parent body
      if (m && sim->pert.select>0) {
        sim->pert.select = m->body_parentid[sim->pert.select];
        sim->pert.skinselect = -1;

        // stop perturbation if world reached
        if (sim->pert.select<=0) {
          sim->pert.active = 0;
        }
      }

      break;

    case ']':                   // cycle up fixed cameras
      if (m && m->ncam) {
        sim->cam.type = mjCAMERA_FIXED;
        // simulate->camera = {0 or 1} are reserved for the free and tracking cameras
        if (sim->camera < 2 || sim->camera == 2 + m->ncam-1) {
          sim->camera = 2;
        } else {
          sim->camera += 1;
        }
        sim->cam.fixedcamid = sim->camera - 2;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case '[':                   // cycle down fixed cameras
      if (m && m->ncam) {
        sim->cam.type = mjCAMERA_FIXED;
        // settings.camera = {0 or 1} are reserved for the free and tracking cameras
        if (sim->camera <= 2) {
          sim->camera = 2 + m->ncam-1;
        } else {
          sim->camera -= 1;
        }
        sim->cam.fixedcamid = sim->camera - 2;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_F6:                   // cycle frame visualisation
      if (m) {
        sim->opt.frame = (sim->opt.frame + 1) % mjNFRAME;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_F7:                   // cycle label visualisation
      if (m) {
        sim->opt.label = (sim->opt.label + 1) % mjNLABEL;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_ESCAPE:          // free camera
      sim->cam.type = mjCAMERA_FREE;
      sim->camera = 0;
      mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      break;

    case '-':                   // slow down
      {
        int numclicks = sizeof(sim->percentRealTime) / sizeof(sim->percentRealTime[0]);
        if (sim->real_time_index < numclicks-1 && !state->shift) {
          sim->real_time_index++;
          sim->speed_changed = true;
        }
      }
      break;

    case '=':                   // speed up
      if (sim->real_time_index > 0 && !state->shift) {
        sim->real_time_index--;
        sim->speed_changed = true;
      }
      break;

    // agent keys
    case mjKEY_ENTER:
      sim->agent->plan_enabled = !sim->agent->plan_enabled;
      break;

    case '\\':
      sim->agent->action_enabled = !sim->agent->action_enabled;
      break;

    case '9':
      sim->agent->visualize_enabled = !sim->agent->visualize_enabled;
      break;
    }

    return;
  }

  // 3D scroll
  if (state->type==mjEVENT_SCROLL && state->mouserect==3 && m) {
    // emulate vertical mouse motion = 2% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -zoom_increment*state->sy, &sim->scn, &sim->cam);

    return;
  }

  // 3D press
  if (state->type==mjEVENT_PRESS && state->mouserect==3 && m) {
    // set perturbation
    int newperturb = 0;
    if (state->control && sim->pert.select>0) {
      // right: translate;  left: rotate
      if (state->right) {
        newperturb = mjPERT_TRANSLATE;
      } else if (state->left) {
        newperturb = mjPERT_ROTATE;
      }

      // perturbation onset: reset reference
      if (newperturb && !sim->pert.active) {
        mjv_initPerturb(m, d, &sim->scn, &sim->pert);
      }
    }
    sim->pert.active = newperturb;

    // handle double-click
    if (state->doubleclick) {
      // determine selection mode
      int selmode;
      if (state->button==mjBUTTON_LEFT) {
        selmode = 1;
      } else if (state->control) {
        selmode = 3;
      } else {
        selmode = 2;
      }

      // find geom and 3D click point, get corresponding body
      mjrRect r = state->rect[3];
      mjtNum selpnt[3];
      int selgeom, selflex, selskin;
      int selbody = mjv_select(m, d, &sim->opt,
                               static_cast<mjtNum>(r.width)/r.height,
                               (state->x - r.left)/r.width,
                               (state->y - r.bottom)/r.height,
                               &sim->scn, selpnt, &selgeom, &selflex, &selskin);

      // set lookat point, start tracking is requested
      if (selmode==2 || selmode==3) {
        // copy selpnt if anything clicked
        if (selbody>=0) {
          mju_copy3(sim->cam.lookat, selpnt);
        }

        // switch to tracking camera if dynamic body clicked
        if (selmode==3 && selbody>0) {
          // mujoco camera
          sim->cam.type = mjCAMERA_TRACKING;
          sim->cam.trackbodyid = selbody;
          sim->cam.fixedcamid = -1;

          // UI camera
          sim->camera = 1;
          mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
        }
      }

      // set body selection
      else {
        if (selbody>=0) {
          // record selection
          sim->pert.select = selbody;
          sim->pert.skinselect = selskin;
          sim->pert.flexselect = selflex;

          // compute localpos
          mjtNum tmp[3];
          mju_sub3(tmp, selpnt, d->xpos+3*sim->pert.select);
          mju_mulMatTVec(sim->pert.localpos, d->xmat+9*sim->pert.select, tmp, 3, 3);
        } else {
          sim->pert.select = 0;
          sim->pert.skinselect = -1;
          sim->pert.flexselect = -1;
        }
      }

      // stop perturbation on select
      sim->pert.active = 0;
    }

    return;
  }

  // 3D release
  if (state->type==mjEVENT_RELEASE && state->dragrect==3 && m) {
    // stop perturbation
    sim->pert.active = 0;

    return;
  }

  // 3D move
  if (state->type==mjEVENT_MOVE && state->dragrect==3 && m) {
    // determine action based on mouse button
    mjtMouse action;
    if (state->right) {
      action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (state->left) {
      action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
      action = mjMOUSE_ZOOM;
    }

    // move perturb or camera
    mjrRect r = state->rect[3];
    if (sim->pert.active)
      mjv_movePerturb(m, d, action, state->dx/r.height, -state->dy/r.height,
                      &sim->scn, &sim->pert);
    else
      mjv_moveCamera(m, action, state->dx/r.height, -state->dy/r.height,
                     &sim->scn, &sim->cam);

    return;
  }

  // Dropped files
  if (state->type == mjEVENT_FILESDROP && state->dropcount > 0) {
    while (sim->droploadrequest.load()) {}
    sim->dropfilename = state->droppaths[0];
    sim->droploadrequest.store(true);
    return;
  }

  // Redraw
  if (state->type == mjEVENT_REDRAW) {
    sim->Render();
    return;
  }
}
}  // namespace

namespace mujoco {
namespace mju = ::mujoco::util_mjpc;

Simulate::Simulate(std::unique_ptr<PlatformUIAdapter> platform_ui,
                   std::shared_ptr<mjpc::Agent> a)
    : platform_ui(std::move(platform_ui)),
      uistate(this->platform_ui->state()),
      agent(std::move(a)) {}

//------------------------------------ apply pose perturbations ------------------------------------
void Simulate::ApplyPosePerturbations(int flg_paused) {
  if (this->m != nullptr) {
    mjv_applyPerturbPose(this->m, this->d, &this->pert, flg_paused);  // move mocap bodies only
  }
}

//----------------------------------- apply force perturbations ------------------------------------
void Simulate::ApplyForcePerturbations() {
  if (this->m != nullptr) {
    mjv_applyPerturbForce(this->m, this->d, &this->pert);
  }
}

//------------------------- Tell the render thread to load a file and wait -------------------------
void Simulate::Load(mjModel* m,
                    mjData* d,
                    std::string displayed_filename,
                    bool delete_old_m_d) {
  this->mnew = m;
  this->dnew = d;
  this->delete_old_m_d = delete_old_m_d;
  this->filename = std::move(displayed_filename);

  {
    std::unique_lock<std::mutex> lock(mtx);
    this->loadrequest = 2;

    // Wait for the render thread to be done loading
    // so that we know the old model and data's memory can
    // be free'd by the other thread (sometimes python)
    cond_loadrequest.wait(lock, [this]() { return this->loadrequest == 0; });
  }
}

//------------------------------------- load mjb or xml model --------------------------------------
void Simulate::LoadOnRenderThread() {
  if (this->delete_old_m_d) {
    // delete old model if requested
    if (this->d) {
      mj_deleteData(d);
    }
    if (this->m) {
      mj_deleteModel(m);
    }
  }

  this->m = this->mnew;
  this->d = this->dnew;

  // 创建仪表盘数据提取器
  if (g_dashboard_extractor) {
    delete g_dashboard_extractor;
  }
  g_dashboard_extractor = new mjpc::DashboardDataExtractor(this->m);

  // re-create scene and context
  mjv_makeScene(this->m, &this->scn, maxgeom);
  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW] = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }
  this->platform_ui->RefreshMjrContext(this->m, 50*(this->font+1));

  // clear perturbation state
  this->pert.active = 0;
  this->pert.select = 0;
  this->pert.skinselect = -1;

  // align and scale view unless reloading the same file
  if (this->filename != this->previous_filename) {
    AlignAndScaleView(this);
    this->previous_filename = this->filename;
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // set window title to model name
  if (this->m->names) {
    char title[200] = "MuJoCo MPC : ";
    mju::strcat_arr(title, this->m->names);
    platform_ui->SetWindowTitle(title);
  }

  // set keyframe range and divisions
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[0] = 0;
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[1] = mjMAX(0, this->m->nkey - 1);
  this->ui0.sect[SECT_SIMULATION].item[5].slider.divisions = mjMAX(1, this->m->nkey - 1);

  // rebuild UI sections
  MakeUiSections(this);

  // full ui update
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  UpdateSettings(this);

  // clear request
  this->loadrequest = 0;
  cond_loadrequest.notify_all();

  // set real time index
  int numclicks = sizeof(this->percentRealTime) / sizeof(this->percentRealTime[0]);
  float min_error = 1e6;
  float desired = mju_log(100*this->m->vis.global.realtime);
  for (int click=0; click < numclicks; click++) {
    float error = mju_abs(mju_log(this->percentRealTime[click]) - desired);
    if (error < min_error) {
      min_error = error;
      this->real_time_index = click;
    }
  }
}

//------------------------------------------- rendering --------------------------------------------

// prepare to render
void Simulate::PrepareScene() {
  // data for FPS calculation
  static std::chrono::time_point<Clock> lastupdatetm;

  // update interval, save update time
  auto tmnow = Clock::now();
  double interval = Seconds(tmnow - lastupdatetm).count();
  interval = mjMIN(1, mjMAX(0.0001, interval));
  lastupdatetm = tmnow;

  // no model: nothing to do
  if (!this->m) {
    return;
  }

  // 更新仪表盘数据
  if (this->m && this->d && g_dashboard_extractor) {
    g_dashboard_extractor->Update(this->d, g_dashboard_data);
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // update watch
  if (this->ui0_enable && this->ui0.sect[SECT_WATCH].state) {
    UpdateWatch(this);
    mjui_update(SECT_WATCH, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update joint
  if (this->ui1_enable && this->ui1.sect[SECT_JOINT].state) {
    mjui_update(SECT_JOINT, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update info text
  if (this->info) {
    UpdateInfoText(this, this->info_title, this->info_content, interval);
  }

  // update control
  if (this->ui1_enable && this->ui1.sect[SECT_CONTROL].state) {
    mjui_update(SECT_CONTROL, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update profiler
  if (this->profiler && this->run) {
    UpdateProfiler(this);
  }

  // update sensor
  if (this->sensor && this->run) {
    UpdateSensor(this);
  }

  // update task
  if (this->ui0_enable && this->ui0.sect[SECT_TASK].state) {
    if (!this->agent->allocate_enabled && this->uiloadrequest.load() == 0) {
      mjui_update(SECT_TASK, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    }
  }

  // update agent
  if (this->ui0_enable && this->ui0.sect[SECT_AGENT].state) {
    mjui_update(SECT_AGENT, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update agent profiler
  if (this->agent->plot_enabled && this->uiloadrequest.load() == 0) {
    this->agent->Plots(this->d, this->run);
  }

  // clear timers once profiler info has been copied
  ClearTimeres(this->d);
}

// render the ui to the window
void Simulate::Render() {
  if (this->platform_ui->RefreshMjrContext(this->m, 50*(this->font+1))) {
    UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // get 3D rectangle and reduced for profiler
  mjrRect rect = this->uistate.rect[3];
  mjrRect smallrect = rect;
  if (this->profiler) {
    smallrect.width = rect.width - rect.width/4;
  }

  // no model
  if (!this->m) {
    // blank screen
    mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

    // label
    if (this->loadrequest) {
      mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, "loading", nullptr,
                  &this->platform_ui->mjr_context());
    } else {
      char intro_message[Simulate::kMaxFilenameLength];
      mju::sprintf_arr(intro_message,
                       "MuJoCo version %s\nDrag-and-drop model file here", mj_versionString());
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, intro_message, 0,
                  &this->platform_ui->mjr_context());
    }

    // show last loading error
    if (this->load_error[0]) {
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->load_error, 0,
                  &this->platform_ui->mjr_context());
    }

    // render uis
    if (this->ui0_enable) {
      mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    }
    if (this->ui1_enable) {
      mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }

    // finalize
    this->platform_ui->SwapBuffers();

    return;
  }

  // visualization
  if (this->uiloadrequest.load() == 0) {
    // task-specific
    if (this->agent->ActiveTask()->visualize) {
      this->agent->ActiveTask()->ModifyScene(this->m, this->d, &this->scn);
    }
    // common to all tasks
    this->agent->ModifyScene(&this->scn);
  }

  // render scene
  mjr_render(rect, &this->scn, &this->platform_ui->mjr_context());

  // 渲染仪表盘（2D覆盖层）
  RenderDashboard(g_dashboard_data, rect, &this->platform_ui->mjr_context());

  // show last loading error
  if (this->load_error[0]) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->load_error, 0,
                &this->platform_ui->mjr_context());
  }

  // make pause/loading label
  std::string pauseloadlabel;
  if (!this->run || this->loadrequest) {
    pauseloadlabel = this->loadrequest ? "loading" : "pause";
  }

  // get desired and actual percent-of-real-time
  float desiredRealtime = this->percentRealTime[this->real_time_index];
  float actualRealtime = 100 / this->measured_slowdown;

  // if running, check for misalignment of more than 10%
  float realtime_offset = mju_abs(actualRealtime - desiredRealtime);
  bool misaligned = this->run && realtime_offset > 0.1 * desiredRealtime;

  // make realtime overlay label
  char rtlabel[30] = {'\0'};
  if (desiredRealtime != 100.0 || misaligned) {
    // print desired realtime
    int labelsize = std::snprintf(rtlabel,
                                  sizeof(rtlabel), "%g%%", desiredRealtime);

    // if misaligned, append to label
    if (misaligned) {
      std::snprintf(rtlabel+labelsize,
                    sizeof(rtlabel)-labelsize, " (%-4.1f%%)", actualRealtime);
    }
  }

  // draw top left overlay
  if (!pauseloadlabel.empty() || rtlabel[0]) {
    std::string newline = !pauseloadlabel.empty() && rtlabel[0] ? "\n" : "";
    std::string topleftlabel = rtlabel + newline + pauseloadlabel;
    mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, smallrect,
                topleftlabel.c_str(), nullptr, &this->platform_ui->mjr_context());
  }

  // show ui 0
  if (this->ui0_enable) {
    mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show ui 1
  if (this->ui1_enable) {
    mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show help
  if (this->help) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content,
                &this->platform_ui->mjr_context());
  }

  // show info
  if (this->info) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->info_title, this->info_content,
                &this->platform_ui->mjr_context());
  }

  // show profiler
  if (this->profiler) {
    ShowProfiler(this, rect);
  }

  // show sensor
  if (this->sensor) {
    ShowSensor(this, smallrect);
  }

  // show agent plots
  if (this->agent->plot_enabled && this->uiloadrequest.load() == 0) {
    this->agent->PlotShow(&smallrect, &this->platform_ui->mjr_context());
  }

  // take screenshot, save to file
  if (this->screenshotrequest.exchange(false)) {
    const unsigned int h = uistate.rect[0].height;
    const unsigned int w = uistate.rect[0].width;
    std::unique_ptr<unsigned char[]> rgb(new unsigned char[3*w*h]);
    if (!rgb) {
      mju_error("could not allocate buffer for screenshot");
    }
    mjr_readPixels(rgb.get(), nullptr, uistate.rect[0], &this->platform_ui->mjr_context());

    // flip up-down
    for (int r = 0; r < h/2; ++r) {
      unsigned char* top_row = &rgb[3*w*r];
      unsigned char* bottom_row = &rgb[3*w*(h-1-r)];
      std::swap_ranges(top_row, top_row+3*w, bottom_row);
    }

    // save as PNG
    const std::string path = GetSavePath("screenshot.png");
    if (!path.empty()) {
      if (lodepng::encode(path, rgb.get(), w, h, LCT_RGB)) {
        mju_error("could not save screenshot");
      } else {
        std::printf("saved screenshot: %s\n", path.c_str());
      }
    }
  }

  // finalize
  this->platform_ui->SwapBuffers();
}

void Simulate::InitializeRenderLoop() {
  // Set timer callback (milliseconds)
  mjcb_time = Timer;

  // init abstract visualization
  mjv_defaultCamera(&this->cam);
  mjv_defaultOption(&this->opt);
  InitializeProfiler(this);
  InitializeSensor(this);

  // make empty scene
  mjv_defaultScene(&this->scn);
  mjv_makeScene(nullptr, &this->scn, maxgeom);
  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW] = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }

  // select default font
  int fontscale = ComputeFontScale(*this->platform_ui);
  this->font = fontscale/50 - 1;

  // make empty context
  this->platform_ui->RefreshMjrContext(nullptr, fontscale);

  // init state and uis
  std::memset(&this->uistate, 0, sizeof(mjuiState));
  std::memset(&this->ui0, 0, sizeof(mjUI));
  std::memset(&this->ui1, 0, sizeof(mjUI));

  auto [buf_width, buf_height] = this->platform_ui->GetFramebufferSize();
  this->uistate.nrect = 1;
  this->uistate.rect[0].width = buf_width;
  this->uistate.rect[0].height = buf_height;

  this->ui0.spacing = mjui_themeSpacing(this->spacing);
  this->ui0.color = mjui_themeColor(this->color);
  this->ui0.predicate = UiPredicate;
  this->ui0.rectid = 1;
  this->ui0.auxid = 0;

  this->ui1.spacing = mjui_themeSpacing(this->spacing);
  this->ui1.color = mjui_themeColor(this->color);
  this->ui1.predicate = UiPredicate;
  this->ui1.rectid = 2;
  this->ui1.auxid = 1;

  // set GUI adapter callbacks
  this->uistate.userdata = this;
  this->platform_ui->SetEventCallback(UiEvent);
  this->platform_ui->SetLayoutCallback(UiLayout);

  // populate uis with standard sections
  this->ui0.userdata = this;
  this->ui1.userdata = this;
  mjui_add(&this->ui0, defFile);
  mjui_add(&this->ui0, this->def_option);
  mjui_add(&this->ui0, this->def_simulation);
  mjui_add(&this->ui0, this->def_watch);
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());

  // set VSync to initial value
  this->platform_ui->SetVSync(this->vsync);
}

void Simulate::RenderLoop() {
  // run event loop
  while (!this->platform_ui->ShouldCloseWindow() && !this->exitrequest.load()) {
    {
      const std::lock_guard<std::mutex> lock(this->mtx);

      // load model (not on first pass, to show "loading" label)
      if (this->loadrequest==1) {
        this->LoadOnRenderThread();
      } else if (this->loadrequest>1) {
        this->loadrequest = 1;
      }

      // poll and handle events
      this->platform_ui->PollEvents();

      // prepare to render
      this->PrepareScene();
    }  // std::lock_guard<std::mutex> (unblocks simulation thread)

    // render while simulation is running
    this->Render();
  }

  this->exitrequest.store(true);

  mjv_freeScene(&this->scn);
}

}  // namespace mujoco
