
export const createExperimentMessage = (experimentId?: string) => ({
    title: "Create new experiment",
    success: `A new experiment with ID ${experimentId} created`,
    fail: "Fail to create new experiment!"
})