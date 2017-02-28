-- Author Alexandre Boulch at ONERA, The French aerospace lab
-- All rights reserved.
--
-- This code is licensed under GPL licensed
--
-- It is based on the code from https://github.com/szagoruyko/wide-residual-networks
-- Wide Residual Networks (BMVC 2016)
-- http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
-------------------------------------------------
-------------------------------------------------
--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
local utils = paths.dofile'utils.lua'

assert(opt and opt.depth)
assert(opt and opt.num_classes)
assert(opt and opt.widen_factor)

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function createModel(opt)
   local depth = opt.depth

   local blocks = {}

   local function wide_basic(nInputPlane, nOutputPlane, stride, block_id)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            --convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
            if block_id ==1 then
                convs:add(conv_layer_s1:clone('weight','bias', 'gradWeight','gradBias'))
            elseif block_id==2 then
                convs:add(conv_layer_s2:clone('weight','bias', 'gradWeight','gradBias'))
            elseif block_id==3 then
                convs:add(conv_layer_s3:clone('weight','bias', 'gradWeight','gradBias'))
            else
                convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
            end
         end
      end

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride, block_id)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride,0))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1, block_id))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)


      conv_layer_s1 = Convolution(nStages[2],nStages[2],3,3,1,1,1,1)
      conv_layer_s2 = Convolution(nStages[3],nStages[3],3,3,1,1,1,1)
      conv_layer_s3 = Convolution(nStages[4],nStages[4],3,3,1,1,1,1)

      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1,1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2,2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2,3)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   -- model:get(1).gradInput = nil

   return model
end

return createModel(opt)
