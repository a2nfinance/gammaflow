import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ExperimentState = {
    experiments: any[],
    currentExperiment: any,
    runs: any[],
}


const initialExperimentState: ExperimentState = {
    experiments: [],
    currentExperiment: {
        tags: [
            {key: "github_repo", value: ""},
            {key: "is_private_repo", value: "0"},
        ]
    },
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
        setCurrentExperiment: (state: ExperimentState, action: PayloadAction<any>) => {
            state.currentExperiment = action.payload
        },
    }
})
export const { setList, setRuns, setCurrentExperiment } = experimentSlice.actions;
export default experimentSlice.reducer;