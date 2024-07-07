import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ExperimentState = {
    experiments: any[],
    currentExperiment: any
}


const initialExperimentState: ExperimentState = {
    experiments: [],
    currentExperiment: null
}

export const experimentSlice = createSlice({
    name: 'experiment',
    initialState: initialExperimentState,
    reducers: {
        setList: (state: ExperimentState, action: PayloadAction<any[]>) => {
            state.experiments = action.payload
        }
    }
})
export const { setList } = experimentSlice.actions;
export default experimentSlice.reducer;