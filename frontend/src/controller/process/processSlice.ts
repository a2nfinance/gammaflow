import {
    createSlice,
    PayloadAction
} from "@reduxjs/toolkit";

export const actionNames = {
    createExperimentAction: "createExperimentAction",
    startTrainingAction: "startTrainingAction",
}


type Processes = {
    [key: string]: boolean
}

const initialState: Processes = {
    startTrainingAction: false,
    createExperimentAction: false
}

export const processesSlice = createSlice({
    name: 'process',
    initialState,
    reducers: {
        updateActionStatus: (state, action: PayloadAction<{ actionName: string, value: boolean }>) => {
            state[action.payload.actionName] = action.payload.value;
        },
    }
})

export const { updateActionStatus } = processesSlice.actions;
export default processesSlice.reducer;