import java.util.Scanner;

public class C
{
	static int N = 100010;
	static int[] cot = new int[N];
	static boolean[] vis = new boolean[N];

	static int[] h = new int[N];
	static int[] des = new int[2 * N], next = new int[2 * N];
	static int idx = 0;

	static int[] ans = new int[N];

	public static void main(String[] args)
	{
		Scanner in = new Scanner(System.in);
		int round = in.nextInt();
		for (int z = 0; z < round; z++)
		{
			int n = in.nextInt();
			for (int i = 1; i <= n; i++)
			{
				h[i] = -1;
				vis[i] = false;
				cot[i] = 0;
			}
			idx = 0;

			boolean flag = true;
			for (int i = 1; i < n; i++)
			{
				int a = in.nextInt();
				int b = in.nextInt();

				des[idx] = b;
				next[idx] = h[a];
				h[a] = idx++;

				des[idx] = a;
				next[idx] = h[b];
				h[b] = idx++;

				if (++cot[a] >= 3 | ++cot[b] >= 3)
					flag = false;
			}

			if (!flag)
				System.out.println(-1);
			else
			{
				int startPoint = -1;
				for (int i = 1; i <= n; i++)
					if (cot[i] == 1)
					{
						startPoint = i;
						break;
					}
				dfs(startPoint, 3);
				for (int i = 1; i < n; i++)
					System.out.print(ans[i] + " ");
				System.out.println();
			}
		}
	}

	private static void dfs(int point, int preLen)
	{
		vis[point] = true;
		for (int i = h[point]; i != -1; i = next[i])
			if (!vis[des[i]])
			{
				ans[i / 2 + 1] = 5 - preLen;
				// System.out.println(point + ">" + des[i] + " " + ans[i / 2 + 1] + " " + (i / 2
				// + 1));
				dfs(des[i], ans[i / 2 + 1]);
			}
	}
}
