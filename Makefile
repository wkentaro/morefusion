all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install_anaconda3:
	@.make/install_anaconda3.sh

install_v-hacd:
	@.make/install_v-hacd.sh

install: install_anaconda3 install_v-hacd
	@.make/install.sh

lint:
	@.make/lint.sh

test:
	@.make/test.sh

check:
	@.make/check.sh
