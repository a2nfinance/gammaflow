import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ExperimentState = {
    experiments: any[],
    currentExperiment: any,
    runs: any[],
}


const initialExperimentState: ExperimentState = {
    experiments: [],
    currentExperiment: null,
    runs: []
}

export const experimentSlice = createSlice({
    name: 'experiment',
    initialState: initialExperimentState,
    reducers: {
        setList: (state: ExperimentState, action: PayloadAction<any[]>) => {
            state.experiments = action.payload
        },
        setRuns:  (state: ExperimentState, action: PayloadAction<any[]>) => {
            state.runs = action.payload
        },
    }
})
export const { setList, setRuns } = experimentSlice.actions;
export default experimentSlice.reducer;