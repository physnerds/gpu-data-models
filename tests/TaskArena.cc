// Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

// Local include(s).
#include "TaskArena.h"

 tbb::task_arena& taskArena() {

      static tbb::task_arena arena( 1, 0 );
      return arena;
   }


