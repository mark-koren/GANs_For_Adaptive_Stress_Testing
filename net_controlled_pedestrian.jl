#Tristan McRae
#AA 290
#Controlling the actions of a pedestrian through a pre-trained neural network
#Winter 2017

using DataFrames

numDrives = 10^4 #edit based on number of trials you want to try:Have been using 10^5 for training, 10^4 for testing
terminalTime = 100 #edit based on max # of steps each trial should take
count = 0
currentTime = 0
maxPedestrianDistance = 10 #distance a pedestrian can go in 1 second
speedConversion = 5 #how many meters a velocity of 1 goes in 1 second


columnNames = ["DriveNum", "x", "y", "v", "xped", "yped", "crash", "rt","Rtotal"]
driveTracking = DataFrame(v1 = [0], v2 = [0], v3 = [0], v4 = [0], v5 = [0], v6=[0], v7=[0], v8=[0.0], v9=[0.0])
names!(driveTracking.colindex, map(parse, columnNames))

path = ".\\data"


W1_ = readtable(path*"\\two_layer_weights_1.csv", header = false)
W2_ = readtable(path*"\\two_layer_weights_2.csv", header = false)
W3_ = readtable(path*"\\two_layer_weights_3.csv", header = false)

b1_ = readtable(path*"\\two_layer_biases_1.csv", header = false)
b2_ = readtable(path*"\\two_layer_biases_2.csv", header = false)
b3_ = readtable(path*"\\two_layer_biases_3.csv", header = false)

#print(b3_)

W1 = array(W1_)
W2 = array(W2_)
W3 = array(W3_)
b1 = array(b1_)
b2 = array(b2_)
b3 = array(b3_)


#Pedestrian Controller-------------------------------------------------------------------------


function available_actions(x, y)
  #Based on the relative position of the pedestrian to the car, this finds
  #all actions available to the pedestrian

  available_delta_xs = []
  available_delta_ys = []
  available_actions = []

  #find available xs and ys
  for i = -5:5
    if (x+i>=0  && x+i<=9)
      push!(available_delta_xs, i)
    end
    if (y+i>=-99  && y+i<=99)
      push!(available_delta_ys, i)
    end
  end

  num_xs = length(available_delta_xs)
  num_ys = length(available_delta_ys)

  #find every combination of available xs and ys
  for i = 1:num_xs
    for j = 1:num_ys
      push!(available_actions, [available_delta_xs[i], available_delta_ys[j]])
    end
  end

  return available_actions
end

function choose_move(v, x, y, available_actions)
  #Picks an action at random from 5 available actions that have the lowest cost
  num_actions = length(available_actions)
  cost = zeros(num_actions+1)
  cost[1] = Inf
  rank_indecies = ones(Int64, 5)
  best_move_index = 1
  second_move_index = 1
  third_move_index = 1
  fourth_move_index = 1
  fifth_move_index = 1

  #ranks top 5 actions
  for i = 2:num_actions+1
    input_vector = transpose(vcat(x, y, v, available_actions[i-1]))
    h1 = *(input_vector,W1) + b1
    h2 = *(h1,W2) + b2
    cost[i] = (*(h2,W3) + b3)[1]
    if (cost[i]<cost[rank_indecies[1]][1])
      rank_indecies[5] = rank_indecies[4]
      rank_indecies[4] = rank_indecies[3]
      rank_indecies[3] = rank_indecies[2]
      rank_indecies[2] = rank_indecies[1]
      rank_indecies[1] = i-1
    elseif (cost[i]<cost[rank_indecies[2]][1])
      rank_indecies[5] = rank_indecies[4]
      rank_indecies[4] = rank_indecies[3]
      rank_indecies[3] = rank_indecies[2]
      rank_indecies[2] = i-1
    elseif (cost[i]<cost[rank_indecies[3]][1])
      rank_indecies[5] = rank_indecies[4]
      rank_indecies[4] = rank_indecies[3]
      rank_indecies[3] = i-1
    elseif (cost[i]<cost[rank_indecies[4]][1])
      rank_indecies[5] = rank_indecies[4]
      rank_indecies[4] = i-1
    elseif (cost[i]<cost[rank_indecies[5]][1])
      rank_indecies[5] = i-1
    end
  end
  choice = rand(1 : 5)
  move_index = rank_indecies[choice]
  return available_actions[move_index]
end

function getAction(v, x, y)
  #println("getAction")
  #uses the current state to find available actions
  actions = available_actions(x, y)
  #uses pretrained neural net to judge which available action produces the highest expected reward
  deltaX, deltaY = choose_move(v, x, y, actions)
  return deltaX, deltaY
end

#Black-Box simulator. Probably best not to touch anything in here---------------------------------------------

function pedestrians(state, deltaX, deltaY)
  #println("pedestrians")
  #srand(seed)
  vxy = stateToValues(state)
  v = vxy[1]
  x = newX(vxy[2], deltaX)
  y = newY(vxy[3], deltaY)
  newState = stateFromValues(v, x, y)
  return newState
end

function newX(x, dx)
  #println("newX")
  if (dx>0) newX = min(x+dx,9) else newX = max(x+dx, 0) end
  return newX
end

function newY(y, dy)
  #println("newY")
  if (dy>0) newY = min(y+dy,99) else newY = max(y+dy, -99) end
  return newY
end

function safe(x, y, v)
  #println("safeDistance")
  safe = y/v > 2 || x > 5
  return safe
end

function carAction(state)
  #println("carAction")
  status = stateToValues(state)
  v = status[1]
  x = status[2]
  y = status[3]
  if (y > 0)
    if (v < 4)
      vNew = v + 1
    else
      vNew = v
    end
  else
    if (safe(x,y,v))
      vNew = v
    else
      vNew = v - 1
    end
  end
  xNew = x
  yNew = div((y + (v+vNew)/2 * speedConversion),1)
  yNew = max(min(yNew, 99),-99)
  newState = stateFromValues(vNew, xNew, yNew)
  return newState
end

function stateFromValues(v, x, y)
  #println("stateFromValues")
  v = round(v)
  x = round(x)
  y = round(y)
  state = abs(v) + 10 * abs(x) + 100 * abs(y)
  if (y >= 0) state = state + 10000 end
  return state
end

function stateToValues(state)
  #println("stateToValues")
  v = state % 10
  x = ((state - v) % 100 ) / 10
  yAbs = ((state - v - 10*x) % 10000) / 100
  positive = div(state, 10000)
  y = yAbs * (-1)^(1-positive)
  return [v, x, y]
end

function initialize()
  #println("initialize")
  #restarts process
  v = 4
  x = 5
  y = -99
  state = stateFromValues(v, x, y)
  return state
end

function nextStep(initialState, deltaX, deltaY)
  #state, deltaX, deltaY
  #println("nextStep")
  #use carSensors and carActions to find out what happens next
  initialStateInfo = stateToValues(initialState)
  #intermediateState = pedestrians(initialState, seed)
  intermediateStateInfo = initialStateInfo + [0, deltaX, deltaY]
  intermediateState = stateFromValues(intermediateStateInfo[1],intermediateStateInfo[2],intermediateStateInfo[3])
  pedx = intermediateStateInfo[2]-initialStateInfo[2]
  pedy = intermediateStateInfo[3]-initialStateInfo[3]
  y1 = initialStateInfo[3]
  p = probability(initialState, intermediateState)
  nextState = carAction(intermediateState)
  nextStateInfo = stateToValues(nextState)
  v = nextStateInfo[1]
  x = nextStateInfo[2]
  y2 = nextStateInfo[3]
  crash = isCrash(x, y1, y2)
  terminal = crash||hazardPassed(y2, v)

  return [p, crash, nextState, terminal, pedx, pedy]
end

function probability(state1, state2)
  #println("probability")
  vxy1 = stateToValues(state1)
  vxy2 = stateToValues(state2)
  x1 = vxy1[2]
  y1 = vxy1[3]
  x2 = vxy2[2]
  y2 = vxy2[3]

  if (x2 == 0)
    #stuck in road
    if (x1 == 0)
      px = 7/12
    else
      px = 0.5 * (6-x1)/6
    end
  elseif (x2 == 9)
    #stuck on far side
    if (x1 == 9)
      px = 7/12
    else
      px = 0.5 * (x1-3)/6
    end
  elseif (x1 == x2)
    #regular transition
    px = 1/6
  else
    px = 1/12
  end

  if (y2 == 99)
    #max car can be past pedestrian
    if (y1 == 99)
      py = 7/12
    else
      py = 0.5 * (y1 - 93)/6
    end
  elseif (y2 == -99)
    #max pedestrian can be in front of car
    if (y1 == -99)
      py = 7/12
    else
      py = 0.5 * (-93 - y1)/6
    end
  elseif (y1 == y2)
    #regular transition
    py = 1/6
  else
    py = 1/12
  end

  p = px * py
  return p
end

function hazardPassed(y, v)
  #println("hazardPassed")
  #checks if you're a safe distance from the pedestrian
  safe = (y >= 30)||(v==0)
  return safe
end

function isCrash(x, y1, y2)
  #println("isCriticalEvent")
  #checks if there was a collision, regardless of speed
  crash = (x==0 && y1 < 0 && y2 >= 0)
  return crash
end

#Reward---------------------------------------------------------


function reward(probability, crash, terminal, dClose)
  #println("reward")
  #gives reward for actions based on probability

  #these constants would have to be totally rethought if pedestrains could do anything really unlikely
  b = 1000 #subtractive constant for crash
  m = 100 #multaplicitive constant for distance

  if (crash == 1)
    r = 0
  elseif (terminal == 1)
    r = log(probability)-b-dClose*m
  else
    r = log(probability)
  end
  return -r   #This would totally screw with the way MCTS works but may be helpful for DL
end
#Main control---------------------------------------------------------------

function basicDrive(driveTracking, driveNum)
  #println("basicDrive")
  #drives through the course once to see what happens
  state = initialize()
  terminal = 0
  r = 0
  xnew= 0
  ynew=0
  vnew=0
  crash=0
  dClose = Inf

  while (terminal ==0)
    vxy = stateToValues(state)
    v = vxy[1]
    x = vxy[2]
    y = vxy[3]
    d = sqrt(x^2+y^2)
    if (d<dClose) dClose = d end
    deltaX, deltaY = getAction(vxy[1], vxy[2], vxy[3])
    newStep = nextStep(state, deltaX, deltaY)
    p = newStep[1]
    crash = newStep[2]
    state = newStep[3]
    terminal = newStep[4]
    xped = newStep[5]
    yped = newStep[6]
    roundReward = reward(p, crash, terminal, dClose)
    r = r + roundReward
    driveTracking = addDriveStep(driveTracking, driveNum, x, y, v, xped, yped, crash, r, 0)
  end
  return [r, crash]
end

function basicExecute(driveTracking)
  #println("basicExecute")
  #drives a defined # of times and records crashes and their probabilities
  driveRewards = ones(numDrives)
  crash = zeros(numDrives)
  crashCount = 0
  println("Starting Simulation")
  for i = 1:numDrives
    if (i%100 == 0) println("Drive Number = ", i) end
    driveRewards[i], crash[i] = basicDrive(driveTracking, i)
    if (crash[i] == 1) crashCount = crashCount + 1 end
  end
  println("Number of crashes = ", crashCount)
  pTotal = 0
  for i = 1:numDrives
    if (crash[i] == 1) pTotal = pTotal + e^driveRewards[i] end
  end
  println("Total Probability = ", pTotal)
  println("Max Probability = ", e^maximum(driveRewards))
  #return [driveRewards]
  return driveTracking
end

function addDriveStep(driveTracking, driveNum, x, y, v, d, theta, crash, rt, Rtotal)
  #println("addDriveStep")
  push!(driveTracking, @data([driveNum, x, y, v, d, theta, crash, rt, Rtotal]))
  return driveTracking
end

function printDrive(driveTracking)
  #println("printDrive")
  numRows = size(driveTracking)[1]
  Rtotal = -Inf
  driveNum = 0
  driveLength = 0

  for i = 2:numRows
    if driveTracking[i, Symbol("DriveNum")] == driveNum    #if this isn't the first step of a new drive
      driveLength = driveLength+1
      Rtotal = driveTracking[i, Symbol("rt")]
    else                                                #if the last drive just ended
      for j = i-driveLength:i-1
        driveTracking[j, Symbol("Rtotal")] = Rtotal       #fill in Rtotal for the previous drive
      end
      driveLength = 1                                   #reset parameters for new drive
      Rtotal = driveTracking[i, Symbol("rt")]
      driveNum = driveTracking[i, Symbol("DriveNum")]
    end
  end
  for j = numRows-driveLength+1:numRows
    driveTracking[j, Symbol("Rtotal")] = Rtotal       #fill in Rtotal for the previous drive
  end
  inputs = driveTracking[:, 2:6]
  labels = driveTracking[:, [:Rtotal]]

  writetable(path*"\\two_layer_generated3_train.csv", driveTracking)
  writetable(path*"\\two_layer_generated3_train_inputs.csv", inputs)
  writetable(path*"\\two_layer_generated3_train_labels.csv", labels)
end

function resetDriveTracking(driveTracking)
  #println("driveTracking")
  columnNames = ["DriveNum", "x", "y", "v", "d", "theta", "crash", "r"]
  driveTracking = DataFrame(v1 = [0], v2 = [0], v3 = [0], v4 = [0], v5 = [0], v6=[0], v7=[0], v8=[0.0])
  names!(driveTracking.colindex, map(parse, columnNames))
  return driveTracking
end
