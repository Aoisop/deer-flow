import { afterEach, beforeEach, expect, test, rs } from "@rstest/core";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

import LoginPage from "@/app/(auth)/login/page";
import type * as AuthProviderModule from "@/core/auth/AuthProvider";
import { useAuth } from "@/core/auth/AuthProvider";
import type * as SetupModule from "@/core/auth/setup";
import { enUS } from "@/core/i18n/locales/en-US";

// The auth API accepts any non-empty address it issued a session for (see merged
// #7449), including a non-ASCII local part. A browser `type="email"` constraint
// rejects 用户@example.com before the handler ever runs, so the account cannot be
// signed into through this form. Real Chrome: `input.validity.typeMismatch === true`.
// The test DOM here is more permissive than the spec, so the submit-firing case below
// is a control and the attribute case is the regression pin.
rs.mock("@/core/auth/AuthProvider", () => ({
  ...(rs.requireActual<typeof AuthProviderModule>(
    "@/core/auth/AuthProvider",
  ) as object),
  useAuth: rs.fn(),
}));
rs.mock("@/core/i18n/hooks", () => ({ useI18n: () => ({ t: enUS }) }));
rs.mock("next/navigation", () => ({
  useRouter: () => ({ push: rs.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));
rs.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", resolvedTheme: "light" }),
}));
rs.mock("@/components/ui/flickering-grid", () => ({
  FlickeringGrid: () => null,
}));
rs.mock("@/core/auth/setup", () => {
  const actual = rs.requireActual<typeof SetupModule>("@/core/auth/setup");
  return {
    ...actual,
    fetchSetupStatus: rs.fn(async () => ({
      needs_setup: false,
      registration_enabled: true,
    })),
  };
});

const BACKEND_VALID_ADDRESS = "用户@example.com";

beforeEach(() => {
  rs.mocked(useAuth).mockReturnValue({
    isAuthenticated: false,
  } as ReturnType<typeof useAuth>);
  globalThis.fetch = rs.fn(
    async () =>
      new Response(
        JSON.stringify({ detail: { code: "invalid_credentials" } }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        },
      ),
  ) as unknown as typeof globalThis.fetch;
});

afterEach(() => {
  cleanup();
  rs.clearAllMocks();
});

function emailField() {
  return screen.getByLabelText<HTMLInputElement>(enUS.login.email);
}

test("the email field does not carry a browser grammar constraint", () => {
  render(<LoginPage />);

  const input = emailField();
  input.value = BACKEND_VALID_ADDRESS;
  // Read through the attributes: this DOM mirrors `autocomplete` onto the
  // attribute but not onto the `autoComplete` IDL attribute.
  expect(input.getAttribute("type")).toBe("text");
  expect(input.getAttribute("inputmode")).toBe("email");
  expect(input.getAttribute("autocomplete")).toBe("username");
  expect(input.required).toBe(true);
  expect(input.validity.typeMismatch).toBe(false);
  expect(input.checkValidity()).toBe(true);
});

test("signing in with that address reaches the auth endpoint", async () => {
  render(<LoginPage />);
  await waitFor(() => screen.getByRole("button", { name: enUS.login.signIn }));

  fireEvent.change(emailField(), {
    target: { value: BACKEND_VALID_ADDRESS },
  });
  fireEvent.change(
    screen.getByLabelText<HTMLInputElement>(enUS.login.password),
    { target: { value: "correct-horse-battery" } },
  );
  fireEvent.click(screen.getByRole("button", { name: enUS.login.signIn }));

  const calls = (
    globalThis.fetch as unknown as {
      mock: { calls: [string, RequestInit][] };
    }
  ).mock.calls;
  // The page fetches the provider list on mount; the login POST is the one that matters.
  const login = calls.find(([url]) => url === "/api/v1/auth/login/local");
  expect(login).toBeTruthy();
  const body = login?.[1].body as URLSearchParams;
  expect(body.get("username")).toBe(BACKEND_VALID_ADDRESS);
});

test("sign-up mode keeps the same unconstrained field", async () => {
  render(<LoginPage />);
  await waitFor(() => screen.getByRole("button", { name: enUS.login.signIn }));

  fireEvent.click(
    screen.getByRole("button", { name: enUS.login.noAccountSignUp }),
  );

  const input = emailField();
  expect(input.type).toBe("text");
  expect(input.validity.typeMismatch).toBe(false);
});
