import java.util.*;

/** 
__                  __
( _)                ( _)
/ / \\              / /\_\_
/ /   \\            / / | \ \
/ /     \\          / /  |\ \ \
/  /   ,  \ ,       / /   /|  \ \
/  /    |\_ /|      / /   / \   \_\
/  /  |\/ _ '_| \   / /   /   \    \\
|  /   |/  0 \0\    / |    |    \    \\
|    |\|      \_\_ /  /    |     \    \\
|  | |/    \.\ o\o)  /      \     |    \\
\    |     /\\`v-v  /        |    |     \\
| \/    /_| \\_|  /         |    | \    \\
| |    /__/_ `-` /   _____  |    |  \    \\
\|    [__]  \_/  |_________  \   |   \    ()
/    [___] (    \         \  |\ |   |   //
|    [___]                  |\| \|   /  |/
/|    [____]                  \  |/\ / / ||
(  \   [____ /     ) _\      \  \    \| | ||
\  \  [_____|    / /     __/    \   / / //
|   \ [_____/   / /        \    |   \/ //
|   /  '----|   /=\____   _/    |   / //
__ /  /        |  /   ___/  _/\    \  | ||
(/-(/-\)       /   \  (/\/\)/  |    /  | /
(/\/\)           /   /   //
_________/   /    /
\____________/    (



*/

public class Main {
	public static void main(String[] args) {
		Scanner sc=new Scanner(System.in);
		int  t=sc.nextInt();
		while(t-- >0) {
			int n=sc.nextInt();
			int[] arr=new int[n];
			for(int i=0;i<n;i++) {
				arr[i]=sc.nextInt();
			}
			String str=sc.next();
			
			ArrayList<Pair> plist=new ArrayList<>();
			for(int i=0;i<n;i++) {
				char ch=str.charAt(i);
				plist.add(new Pair(arr[i],ch));
			}
			
			//B-reduce
			//R-increse
			Collections.sort(plist);
			
			int counter=1;
			boolean flag=false;
			for(int i=0;i<plist.size();i++) {
				int val=plist.get(i).number;
				int clr=plist.get(i).color;
				
				if(clr=='B') {
					if(val<counter) {
						flag=true;
						break;
					}
				}
				else {
					if(val>counter) {
						flag=true;
						break;
					}
					
				}
				counter++;
			}
			System.out.println(flag?"NO":"YES");
			
			
		}
	}
	public static class Pair implements Comparable<Pair>{
		int number;
		char color;
		
		Pair(int number,char color){
			this.number=number;
			this.color=color;
		}
		
		
		
		
	}
}
