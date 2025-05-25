#ifndef BAER_ETHERCAT_ETHERCATSLAVEBASE_H
#define BAER_ETHERCAT_ETHERCATSLAVEBASE_H

#include <string>

class EthercatSlaveBase {
public:
    EthercatSlaveBase(const std::string& name, uint32_t address);
    virtual ~EthercatSlaveBase() = default;

    virtual void updateRead() = 0;

    virtual void updateWrite() = 0;

    uint32_t get_address() const { return address_; }

    std::string get_name() const { return slave_name_;}
protected:
    void printWarnNotImplemented() { printf("Functionality is not implemented.\n"); }

    // The bus address
    uint32_t address_{0};
    std::string slave_name_;
};


#endif //BAER_ETHERCAT_ETHERCATSLAVEBASE_H
