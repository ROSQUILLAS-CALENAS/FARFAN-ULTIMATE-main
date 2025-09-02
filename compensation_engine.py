"""
Distributed compensation system for handling distributed transactions.
Implements the Saga pattern with support for automatic compensations and recovery.
"""

import ast
import asyncio
import json
import logging
import operator
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from pydantic import field_validator

logger = logging.getLogger(__name__)


# Custom exception hierarchy
class CompensationError(Exception):
    """Base exception for compensation-related errors"""

    def __init__(
        self, message: str, context: Dict[str, Any] = None, cause: Exception = None
    ):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()


class SagaError(CompensationError):
    """Exception for saga pattern execution errors"""

    pass


class ValidationError(CompensationError):
    """Exception for input validation errors"""

    pass


class CompensationHandlerError(CompensationError):
    """Exception for compensation handler execution errors"""

    pass


class ConditionEvaluationError(CompensationError):
    """Exception for condition evaluation errors"""

    pass


# Safe AST-based expression evaluator
class SafeExpressionEvaluator:
    """Safe expression evaluator using AST to prevent code injection"""

    # Allowed operators for safe evaluation
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Allowed functions for safe evaluation
    SAFE_FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

    @classmethod
    def evaluate(cls, expression: str, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate an expression using AST parsing

        Args:
            expression: The expression to evaluate
            context: Variable context for evaluation

        Returns:
            The result of the expression

        Raises:
            ConditionEvaluationError: If expression is invalid or unsafe
        """
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode="eval")
            return cls._eval_node(node.body, context)
        except (SyntaxError, ValueError) as e:
            raise ConditionEvaluationError(
                f"Invalid expression syntax: {expression}",
                context={"expression": expression, "error": str(e)},
                cause=e,
            )
        except Exception as e:
            raise ConditionEvaluationError(
                f"Expression evaluation failed: {expression}",
                context={
                    "expression": expression,
                    "context_keys": list(context.keys()),
                },
                cause=e,
            )

    @classmethod
    def _eval_node(cls, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Recursively evaluate AST nodes"""

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        elif isinstance(node, ast.NameConstant):  # Python < 3.8 compatibility
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            elif node.id in cls.SAFE_FUNCTIONS:
                return cls.SAFE_FUNCTIONS[node.id]
            else:
                raise NameError(f"Name '{node.id}' is not defined or not allowed")
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left, context)
            right = cls._eval_node(node.right, context)
            if type(node.op) in cls.SAFE_OPERATORS:
                return cls.SAFE_OPERATORS[type(node.op)](left, right)
            else:
                raise ValueError(f"Operator {type(node.op)} is not allowed")
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand, context)
            if type(node.op) in cls.SAFE_OPERATORS:
                return cls.SAFE_OPERATORS[type(node.op)](operand)
            else:
                raise ValueError(f"Unary operator {type(node.op)} is not allowed")
        elif isinstance(node, ast.Compare):
            left = cls._eval_node(node.left, context)
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                right = cls._eval_node(comparator, context)
                if type(op) in cls.SAFE_OPERATORS:
                    result = cls.SAFE_OPERATORS[type(op)](left, right)
                    if not result:
                        return False
                    left = right  # For chained comparisons
                else:
                    raise ValueError(f"Comparison operator {type(op)} is not allowed")
            return True
        elif isinstance(node, ast.BoolOp):
            values = [cls._eval_node(value, context) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
            else:
                raise ValueError(f"Boolean operator {type(node.op)} is not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in cls.SAFE_FUNCTIONS:
                func = cls.SAFE_FUNCTIONS[node.func.id]
                args = [cls._eval_node(arg, context) for arg in node.args]
                kwargs = {
                    kw.arg: cls._eval_node(kw.value, context) for kw in node.keywords
                }
                return func(*args, **kwargs)
            else:
                raise ValueError(f"Function calls not allowed or function not safe")
        elif isinstance(node, ast.List):
            return [cls._eval_node(element, context) for element in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(cls._eval_node(element, context) for element in node.elts)
        elif isinstance(node, ast.Dict):
            keys = [cls._eval_node(k, context) for k in node.keys]
            values = [cls._eval_node(v, context) for v in node.values]
            return dict(zip(keys, values))
        else:
            raise ValueError(f"AST node type {type(node)} is not allowed")


# Pydantic models for validation
class CompensationHandlerConfig(BaseModel):
    """Configuration for compensation handlers"""

    handler_name: str = Field(..., min_length=1, max_length=100)
    handler_func: Callable
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("handler_name")
    @classmethod
    def validate_handler_name(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Handler name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


class CompensationAction(BaseModel):
    """Represents a compensation action to be executed"""

    action_id: str = Field(..., min_length=1)
    workflow_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    handler_name: str = Field(..., min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None
    priority: int = Field(default=0, ge=0, le=100)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Condition must be a string expression")
        if v and len(v.strip()) == 0:
            raise ValueError("Condition cannot be empty string")
        return v


class CompensationConfig(BaseModel):
    """Configuration for compensation engine"""

    max_parallel_compensations: int = Field(default=5, ge=1, le=20)
    default_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    max_retry_attempts: int = Field(default=3, ge=0, le=10)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    enable_condition_evaluation: bool = Field(default=True)


class WorkflowCompensationRequest(BaseModel):
    """Request for workflow compensation"""

    workflow_id: str = Field(..., min_length=1, max_length=100)
    execution_id: Optional[str] = None
    force_compensation: bool = Field(default=False)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=7200)

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Workflow ID must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


class CompensationEngine:
    """
    Handles automatic compensation and rollback for failed workflows with enhanced security and validation
    """

    def __init__(self, event_bus=None, config: CompensationConfig = None):
        """
        Initialize compensation engine with validation

        Args:
            event_bus: Event bus for publishing events
            config: Configuration for the compensation engine

        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            self.config = config or CompensationConfig()
            self.event_bus = event_bus
            self.compensation_handlers: Dict[str, CompensationHandlerConfig] = {}
            self.active_compensations: Dict[str, List[CompensationAction]] = {}
            self.evaluator = SafeExpressionEvaluator()
            self._compensation_lock = asyncio.Lock()

            logger.info("CompensationEngine initialized successfully")

        except Exception as e:
            raise ValidationError(
                "Failed to initialize CompensationEngine",
                context={"config": config.dict() if config else None},
                cause=e,
            )

    def register_compensation_handler(self, config: CompensationHandlerConfig):
        """
        Register a compensation handler with full validation

        Args:
            config: Handler configuration with validation

        Raises:
            ValidationError: If configuration is invalid
            CompensationHandlerError: If handler registration fails
        """
        try:
            # Validate configuration
            validated_config = CompensationHandlerConfig.parse_obj(config)

            if validated_config.handler_name in self.compensation_handlers:
                raise CompensationHandlerError(
                    f"Handler already registered: {validated_config.handler_name}",
                    context={"handler_name": validated_config.handler_name},
                )

            # Validate handler is callable
            if not callable(validated_config.handler_func):
                raise ValidationError(
                    f"Handler function must be callable: {validated_config.handler_name}",
                    context={"handler_name": validated_config.handler_name},
                )

            self.compensation_handlers[validated_config.handler_name] = validated_config
            logger.info(
                f"Registered compensation handler: {validated_config.handler_name}"
            )

        except ValidationError as e:
            raise e
        except Exception as e:
            raise CompensationHandlerError(
                f"Failed to register compensation handler",
                context={
                    "config": config.dict() if hasattr(config, "dict") else str(config)
                },
                cause=e,
            )

    async def compensate_workflow(self, request: WorkflowCompensationRequest) -> bool:
        """
        Execute compensation for a failed workflow with comprehensive validation and error handling

        Args:
            request: Validated compensation request

        Returns:
            True if compensation succeeded, False otherwise

        Raises:
            ValidationError: If request is invalid
            SagaError: If compensation execution fails
        """
        try:
            # Validate request
            validated_request = WorkflowCompensationRequest.parse_obj(request)

            async with self._compensation_lock:
                workflow_id = validated_request.workflow_id

                if workflow_id not in self.active_compensations:
                    logger.warning(
                        f"No compensations found for workflow: {workflow_id}"
                    )
                    return True

                compensations = self.active_compensations[workflow_id]

                if not compensations:
                    logger.info(f"No compensation actions for workflow: {workflow_id}")
                    return True

                # Sort by priority (higher priority first) then by creation time (reverse order)
                sorted_compensations = sorted(
                    compensations,
                    key=lambda x: (-x.priority, x.created_at),
                    reverse=True,
                )

                success_count = 0
                failure_count = 0

                for compensation in sorted_compensations:
                    try:
                        success = await self._execute_compensation(compensation)
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
                    except CompensationHandlerError as e:
                        logger.error(f"Compensation handler failed: {e}")
                        failure_count += 1
                    except Exception as e:
                        logger.error(f"Unexpected error during compensation: {e}")
                        failure_count += 1

                # Clean up completed compensations
                del self.active_compensations[workflow_id]

                total_compensations = success_count + failure_count
                success_rate = (
                    success_count / total_compensations
                    if total_compensations > 0
                    else 1.0
                )

                logger.info(
                    f"Workflow compensation completed: {workflow_id}, "
                    f"Success: {success_count}, Failures: {failure_count}, "
                    f"Success rate: {success_rate:.2%}"
                )

                # Consider compensation successful if at least 80% of actions succeeded
                return success_rate >= 0.8

        except ValidationError as e:
            raise e
        except Exception as e:
            raise SagaError(
                f"Compensation failed for workflow",
                context={
                    "workflow_id": getattr(request, "workflow_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

    async def _execute_compensation(self, compensation: CompensationAction) -> bool:
        """
        Execute a single compensation action with enhanced error handling

        Args:
            compensation: The compensation action to execute

        Returns:
            True if compensation succeeded, False otherwise

        Raises:
            CompensationHandlerError: If handler execution fails critically
        """
        try:
            # Validate compensation action
            validated_compensation = CompensationAction.parse_obj(compensation)

            # Check if handler exists
            if validated_compensation.handler_name not in self.compensation_handlers:
                raise CompensationHandlerError(
                    f"Compensation handler not found: {validated_compensation.handler_name}",
                    context={
                        "handler_name": validated_compensation.handler_name,
                        "available_handlers": list(self.compensation_handlers.keys()),
                    },
                )

            handler_config = self.compensation_handlers[
                validated_compensation.handler_name
            ]

            # Evaluate condition if present
            if (
                validated_compensation.condition
                and self.config.enable_condition_evaluation
            ):
                try:
                    should_execute = self.evaluator.evaluate(
                        validated_compensation.condition, validated_compensation.context
                    )
                    if not should_execute:
                        logger.info(
                            f"Skipping compensation due to condition: {validated_compensation.condition}"
                        )
                        return True  # Consider skipped as success
                except ConditionEvaluationError as e:
                    logger.error(f"Condition evaluation failed: {e}")
                    # Continue with execution if condition evaluation fails

            # Execute handler with timeout and retries
            for attempt in range(handler_config.retry_attempts + 1):
                try:
                    # Execute with timeout
                    handler_func = handler_config.handler_func

                    if asyncio.iscoroutinefunction(handler_func):
                        result = await asyncio.wait_for(
                            handler_func(validated_compensation.context),
                            timeout=validated_compensation.timeout_seconds,
                        )
                    else:
                        # Run sync function in thread pool
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, handler_func, validated_compensation.context
                            ),
                            timeout=validated_compensation.timeout_seconds,
                        )

                    logger.info(
                        f"Compensation executed successfully: {validated_compensation.action_id}"
                    )
                    return True

                except asyncio.TimeoutError as e:
                    error_msg = (
                        f"Compensation timed out after {validated_compensation.timeout_seconds}s: "
                        f"{validated_compensation.action_id}"
                    )
                    logger.warning(error_msg)

                    if attempt == handler_config.retry_attempts:
                        raise CompensationHandlerError(
                            error_msg,
                            context={
                                "action_id": validated_compensation.action_id,
                                "timeout": validated_compensation.timeout_seconds,
                                "attempt": attempt + 1,
                            },
                            cause=e,
                        )

                except Exception as e:
                    error_msg = (
                        f"Compensation handler failed (attempt {attempt + 1}): "
                        f"{validated_compensation.action_id} - {str(e)}"
                    )
                    logger.warning(error_msg)

                    if attempt == handler_config.retry_attempts:
                        raise CompensationHandlerError(
                            f"All retry attempts exhausted for compensation: {validated_compensation.action_id}",
                            context={
                                "action_id": validated_compensation.action_id,
                                "attempts": attempt + 1,
                                "final_error": str(e),
                            },
                            cause=e,
                        )

                    # Exponential backoff
                    delay = handler_config.retry_delay_seconds * (
                        self.config.backoff_multiplier**attempt
                    )
                    await asyncio.sleep(delay)

            return False

        except ValidationError as e:
            raise CompensationHandlerError(
                "Invalid compensation action",
                context={
                    "compensation": compensation.dict()
                    if hasattr(compensation, "dict")
                    else str(compensation)
                },
                cause=e,
            )
        except (CompensationHandlerError, ConditionEvaluationError) as e:
            raise e
        except Exception as e:
            raise CompensationHandlerError(
                "Unexpected error during compensation execution",
                context={
                    "action_id": getattr(compensation, "action_id", "unknown"),
                    "handler_name": getattr(compensation, "handler_name", "unknown"),
                },
                cause=e,
            )

    def add_compensation_action(self, action: CompensationAction):
        """
        Add a compensation action for a workflow with validation

        Args:
            action: The compensation action to add

        Raises:
            ValidationError: If action is invalid
        """
        try:
            validated_action = CompensationAction.parse_obj(action)

            workflow_id = validated_action.workflow_id
            if workflow_id not in self.active_compensations:
                self.active_compensations[workflow_id] = []

            self.active_compensations[workflow_id].append(validated_action)

            logger.debug(
                f"Added compensation action: {validated_action.action_id} for workflow: {workflow_id}"
            )

        except Exception as e:
            raise ValidationError(
                "Failed to add compensation action",
                context={
                    "action": action.dict() if hasattr(action, "dict") else str(action)
                },
                cause=e,
            )

    def get_compensation_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get compensation status for a workflow with validation

        Args:
            workflow_id: ID of the workflow

        Returns:
            Dictionary containing compensation status

        Raises:
            ValidationError: If workflow_id is invalid
        """
        try:
            if not isinstance(workflow_id, str) or not workflow_id.strip():
                raise ValidationError(
                    "Workflow ID must be a non-empty string",
                    context={"workflow_id": workflow_id},
                )

            compensations = self.active_compensations.get(workflow_id, [])

            return {
                "workflow_id": workflow_id,
                "total_compensations": len(compensations),
                "pending_compensations": len([c for c in compensations]),
                "registered_handlers": list(self.compensation_handlers.keys()),
                "engine_config": self.config.dict(),
            }

        except ValidationError as e:
            raise e
        except Exception as e:
            raise CompensationError(
                "Failed to get compensation status",
                context={"workflow_id": workflow_id},
                cause=e,
            )
